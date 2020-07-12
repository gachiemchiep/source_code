import tvm
from tvm import te
import time

from matplotlib import pyplot as plt
from tvm.relay.testing.config import ctx_list
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils

supported_model = [
    'ssd_512_resnet50_v1_voc',
    'ssd_512_resnet50_v1_coco',
    'ssd_512_resnet101_v2_voc',
    'ssd_512_mobilenet1.0_voc',
    'ssd_512_mobilenet1.0_coco',
    'ssd_300_vgg16_atrous_voc'
    'ssd_512_vgg16_atrous_coco',
]

model_name = supported_model[0]
dshape = (1, 3, 512, 512)
target = "cuda"
# target = "llvm"
ctx = tvm.context(target)

######################################################################
# Download and pre-process demo image

im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                             'gluoncv/detection/street_small.jpg?raw=true',
                             'street_small.jpg', module='data')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

######################################################################
# Convert and compile model
block = model_zoo.get_model(model_name, pretrained=True)

def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
    return graph, lib, params

######################################################################
# Create TVM runtime and do inference

def run(graph, lib, params, ctx):
    # Build TVM runtime
    m = graph_runtime.create(graph, lib, ctx)
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input('data', tvm_input)
    m.set_input(**params)

    run_cnt = 1000
    start_time = time.time()
    for i in range(run_cnt):
        # execute
        m.run()
        # get outputs
        class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    end_time = time.time()
    inference_time = (end_time - start_time) / float(run_cnt) * 1000 # ms
    print("Inference time: {:6.1f} ms".format(inference_time))
    return class_IDs, scores, bounding_boxs

print("TARGET: {}".format(target))
graph, lib, params = build(target)
class_IDs, scores, bounding_boxs = run(graph, lib, params, ctx)

######################################################################
# Display result

# ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
#                          class_IDs.asnumpy()[0], class_names=block.classes)
# plt.show()
