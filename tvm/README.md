


Classification : gluoncv

The tvm benchmark result is at : https://github.com/apache/incubator-tvm/wiki/Benchmark#nvidia-gpu


| Framework | Model                   |   CPU   |    GPu |
| --------- | ----------------------- | :-----: | -----: |
| gluoncv   | resnet18_v1             | 5.4 ms  | 5.0 ms |
| tvm       | resnet18_v1             | 37.6 ms | 2.7 ms |
| gluoncv   | ssd_512_resnet50_v1_voc |  10.7 ms       |    7.4 ms    |
| tvm       | ssd_512_resnet50_v1_voc |         |        |

ssd_512_resnet50_v1_voc

```bash
-> so tvm did accelerate speed of GPu inferenceing

# model : ssd_512_resnet50_v1_voc
# gluoncv  
Inference time:   10.3 ms
# tvm 
Inference time:  634.0 ms


????

# Huhm : look like we need to do the auto tune on GPU first
# then use the auto-tuned image for inferencing

# https://github.com/apache/incubator-tvm/issues/3912
```

No speed increase when converting Insightface (MXNET) model to TVM #3912
    https://github.com/apache/incubator-tvm/issues/3912


tune_conv2d_cuda.py 
    Finding tvm configuration for 1 layer 
    Run for 1000 times and we can get a config that allow faster fowarding

    Run the tunning
    save the result in conv2d.log
    Then use the best config of conv2d.log

    Best config:
    [('tile_f', [-1, 2, 8, 1]), ('tile_y', [-1, 1, 7, 1]), ('tile_x', [-1, 7, 1, 1]), ('tile_rc', [-1, 8, 1]), ('tile_ry', [-1, 3, 1]), ('tile_rx', [-1, 1, 1]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],None,1947908
    Finish loading 1020 records
    Time cost of this operator: 0.000192

So the workflow of tvm should be as

    1. Use auto-tune to compile the model : this will take a long long time
    2. Then load the compiled model

Some performance benchmark

    https://github.com/apache/incubator-tvm/tree/master/apps/benchmark

doc at : https://tvm.apache.org/docs/

The config (.log) will be upload at : https://github.com/uwsampl/tophub/issues/3
    https://github.com/uwsampl/tophub/issues/3

    -> important point
    nnvm will download them to ~/.tvm/tophub automatically during compilation


Make your update to the existing file, save it with the name of the next version
(i.e. Update v0.05.log and save it to v0.06.log)
Update PACKAGE_VERSION in tophub.py in the TVM repo to use the new file.

need to read and understand tvm first
    https://tvm.apache.org/docs/dev/runtime.html

## Refenrece

1. GPU 対応した AutoTVM を試す: https://qiita.com/masahi/items/dc4a9d74e5cf2c345bdf : 2018
2. TVM の紹介 : https://www.slideshare.net/masahi129/tvm-122375943 : 2018
3. Automatic Kernel Optimization for Deep Learning on All Hardware Platforms : https://tvm.apache.org/2018/10/03/auto-opt-all

