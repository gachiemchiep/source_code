import os

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
from gluoncv.model_zoo import get_model

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""

    # change for cifar
    input_shape = (batch_size, 3, 32, 32)
    output_shape = (batch_size, 10)

    print("Use : {}".format(name))

    if name == "cifar_resnet20_v1":
        input_shape = (batch_size, 3, 32, 32)
        output_shape = (batch_size, 10)

        block = get_model('cifar_resnet20_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)

    elif name == 'ssd_512_resnet50_v1_voc':
        input_shape = (batch_size, 3, 512, 512)
        output_shape = (batch_size, 20)
        block = get_model('ssd_512_resnet50_v1_voc', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)

    elif name == 'hrnet_bottom_up':
        import sys
        sys.path.append("/datadrive/workspace/github/HRNet-Bottom-Up-Pose-Estimation")
        sys.path.append("/datadrive/workspace/github/HRNet-Bottom-Up-Pose-Estimation/lib")

        import sys
        import cv2
        import torch
        from config import cfg, update_config
        import argparse
        import models

        parser = argparse.ArgumentParser(description='Train keypoints network')
        # general
        parser.add_argument('--cfg', type=str, default="/datadrive/workspace/github/HRNet-Bottom-Up-Pose-Estimation/experiments/inference_demo.yaml")
        parser.add_argument('--videoFile', type=str, required=False)
        parser.add_argument('--outputDir', type=str, default='/output/')
        parser.add_argument('--inferenceFps', type=int, default=10)
        parser.add_argument('--visthre', type=float, default=0)
        parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
        args = parser.parse_args()

        update_config(cfg, args)

        input_shape = (1, 3, 512, 512)
        output_shape = None

        cfg.defrost()
        cfg.TEST.MODEL_FILE = "/datadrive/workspace/github/HRNet-Bottom-Up-Pose-Estimation/model/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth"
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        cfg.freeze()

        pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False)
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)

        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(pose_model, input_data).eval()

        mod, params = relay.frontend.from_pytorch(scripted_model, input_shapes=[('data', input_shape)], default_dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)

    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
# network = 'ssd_512_resnet50_v1_voc'
network = 'hrnet_bottom_up'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        # runner=autotvm.RPCRunner(
        #     '1060ti',  # change the device key to your key
        #     '0.0.0.0', 9190,
        #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)


