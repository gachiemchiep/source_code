Run both on GPU:
    Detector: 16.79 ms
    Pose: 227.68 ms

Detector on GPU, Pose on CPU
    Detector: 16.80 ms
    Pose: 952.32 ms


Run both on GPU : without the post processing

Detector: 15.57 ms
Pose: 26.63 ms

# Topdown approach

## AlphaPose : gluoncv 

処理時間のbenchmark

```bash

# 0.09 ms
x_ctx = x.as_in_context(ctx)

# 16.09 ms
class_IDs, scores, bounding_boxs = detector(x_ctx)

# 20.87 ms 
pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs, ctx=ctx)

# 0.09 ms
pose_input = pose_input.as_in_context(ctx)

# 26.48 ms
predicted_heatmap = pose_net(pose_input)

# 232.57 ms
pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

```

結論： post-processing 処理時間は長い、tvmなどのcompiler使っても、早くなりません。

## AlphaPose : github 

```bash

python scripts/demo_inference.py --cfg ${cfg_file} --checkpoint ${trained_model} --indir ${img_directory} --outdir ${output_directory}


python demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml  --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --outdir examples/res 

python demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml  --checkpoint pretrained_models/fast_res50_256x192.pth --video examples/girl_dancing_30secs.mp4 --outdir examples/res --save_video --gpus 0 --vis_fast

```

-> 実行時に、PCが止まりました。

## mmlab の mmpose

* Top down approach

```bash
# install https://github.com/open-mmlab/mmpose/blob/master/docs/install.md
# pose model zoo 
# https://mmpose.readthedocs.io/en/latest/model_zoo.html#bottom-up-method
# 3 person inside frame

# pose_scnet_50
python demo/video_demo_with_mmdet.py     ../mmdetection/configs/ssd/ssd300_coco.py     ../checkpoints/mmdetection/ssd/ssd300_coco_20200307-a92d2092.pth     configs/top_down/scnet/coco/scnet50_coco_256x192.py     ../checkpoints/mmpose/top-down/scnet50_coco_256x192-6920f829_20200709.pth     --video-path ../../examples/girl_dancing_30secs.mp4  --show --out-video-root ./

# Result 
Detection: 35.44 ms
Pose: 107.07 ms
Pose vis: 62.60 ms

# pose_hrnet_w32 256x192: .. 
python demo/video_demo_with_mmdet.py     ../mmdetection/configs/ssd/ssd300_coco.py     ../checkpoints/mmdetection/ssd/ssd300_coco_20200307-a92d2092.pth configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py     ../checkpoints/mmpose/top-down/hrnet_w32_coco_256x192-c78dce93_20200708.pth     --video-path ../../examples/girl_dancing_30secs.mp4  --show --out-video-root ./

# Result:
Detection: 37.00 ms
Pose: 238.11 ms
Pose vis: 66.78 ms

# pose_resnet_50 256x192
python demo/video_demo_with_mmdet.py     ../mmdetection/configs/ssd/ssd300_coco.py     ../checkpoints/mmdetection/ssd/ssd300_coco_20200307-a92d2092.pth configs/top_down/resnet/coco/res50_coco_256x192.py     ../checkpoints/mmpose/top-down/res50_coco_256x192-ec54d7f3_20200709.pth      --video-path ../../examples/girl_dancing_30secs.mp4  --show --out-video-root ./

# Result 
Detection: 35.22 ms
Pose: 60.78 ms
Pose vis: 60.72 ms

```

結論：top-downは精度がたかいですが、処理時間が長い

* bottom-up approach : まだ対応しません、そろそろ

[github issue](https://github.com/open-mmlab/mmpose/issues/31#issuecomment-663337636)


----------> TODO : can TVM be used in the mmpose


## hrnet bottom up 

https://github.com/HRNet/HRNet-Bottom-Up-Pose-Estimation


```bash
python tools/inference_demo.py --cfg experiments/inference_demo.yaml \
    --videoFile ../AlphaPose/examples/girl_dancing_30secs.mp4  \
    --outputDir output \
    --visthre 0.3 \
    TEST.MODEL_FILE model/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth 

# One image is about 330ms
#   Network inference = 290ms
#   Other = 40ms

```

-> can we use tvm to fasten this???

```bash
# See : https://discuss.tvm.ai/t/relay-from-tensorflow-failed-couldnt-find-antlr-parser/4529/9
sudo apt install antlr4
pip install antlr4-python3-runtime


(py3) gachiemchiep:tvm$ python gcv_tune_relay_cuda.py 
Extract tasks...
Use : hrnet_bottom_up
=> loading model from /datadrive/workspace/github/HRNet-Bottom-Up-Pose-Estimation/model/pose_coco/pose_hrnet_w32_reg_delaysep_bg01_stn_512_adam_lr1e-3_coco_x140.pth
Traceback (most recent call last):
  File "gcv_tune_relay_cuda.py", line 218, in <module>
    tune_and_evaluate(tuning_option)
  File "gcv_tune_relay_cuda.py", line 180, in tune_and_evaluate
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
  File "gcv_tune_relay_cuda.py", line 84, in get_network
    scripted_model = torch.jit.trace(pose_model, input_data).eval()
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 686, in trace
    traced = _module_class(func, **executor_options)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1482, in __init__
    self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1482, in __init__
    self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1482, in __init__
    self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1482, in __init__
    self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1482, in __init__
    self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1046, in init_then_register
    original_init(self, *args, **kwargs)
  File "/home/gachiemchiep/miniconda3/envs/py3/lib/python3.7/site-packages/torch/jit/__init__.py", line 1456, in __init__
    assert(isinstance(orig, torch.nn.Module))
AssertionError

```

-> 結論: hrnet bottom up は pytorch 1.1 を利用しています。なにかの原因で上記のエラーが出てきました。

hrnet bottom up は mmposeの中に利用されるので、mmposeのVersionが更新された後に再会します。

https://github.com/open-mmlab/mmpose/issues/9

## openvino 

There's demo video at : https://www.youtube.com/watch?v=238KPQUgQxI
    human-pose-estimation-0001 
-> barely archieve 9fps


-------------> 
結論：　最新のpose proposal はまだ早くない
        ー> Realtime 
