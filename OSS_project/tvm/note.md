




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


The config (.log) will be upload at : https://github.com/uwsampl/tophub/issues/3
    https://github.com/uwsampl/tophub/issues/3

    -> important point
    nnvm will download them to ~/.tvm/tophub automatically during compilation


Make your update to the existing file, save it with the name of the next version
(i.e. Update v0.05.log and save it to v0.06.log)
Update PACKAGE_VERSION in tophub.py in the TVM repo to use the new file.

Try tu run tune_relay_cuda.py for cifar_resnet20_v1

    # 0 : let the OS do scheduling
    # https://github.com/apache/incubator-tvm/pull/4145
    export TVM_BIND_THREADS=0

    # use 8 threads
    export TVM_NUM_THREADS=10
    export TVM_BIND_THREADS=1 

see this : https://discuss.tvm.ai/t/auto-tuning-too-slow/4439/13

about : https://tvm.apache.org/about
https://user-images.githubusercontent.com/7287321/51450990-080a1f80-1d6e-11e9-8b1a-bb3f1c03be33.png


some good points 


```bash
https://tvm.apache.org/2017/10/06/nnvm-compiler-announcement

We take a different approach from existing deep learning frameworks, which packages the graph optimization with the deployment runtime

```

The usage of tvm can be summary as 

1. 


## Refenrece


