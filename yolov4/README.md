# Re-produce inferecing speed of Yolov4 using acceleration framework

Detail steps to re-produce the inference speed of Yolov4 reported at https://github.com/AlexeyAB/darknet for the following framework


- [x] tkDNN
- [x] OpenCV
- [ ] TVM : not available yet

## tkDNN

### Prepare

```bash
/datadrive/workspace/tkDNN
├── darknet                 : customed darknet version of tkDNN
├── data                    : where to store yolov4 weight and configure files
    ├── yolov4
        ├── debug
        ├── layers
        ├── yolov4.cfg
        └── yolov4.weights
├── tkDNN                   : tkDNN source code
└── tkDNN.build             : build directory of tkDNN

```

Go to [tkDNN github](https://github.com/ceccocats/tkDNN) and follow [compiling](https://github.com/ceccocats/tkDNN#how-to-compile-this-repo) to compile tkDNN. 


```bash

# Download tkDNN and fullfill the requirements

# Add custom build location of opencv
# Edit CMakeLists.txt
set(OpenCV_DIR ~/opt/opencv/opencv.build/)
find_package(OpenCV REQUIRED PATHS "~/opt/opencv/opencv.build/")

# then compile 
mkdir tkDNN.build;cd  tkDNN.build
cmake ../tkDNN
make
```

### Testing with Yolov4 (608x608)

The model *YOLOv4 (608x608): is used to test. Download [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) , [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg), [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names) . 

(Note: tkDNN use the yolov4 416x416 by default)

```bash
├── data                    
    ├── yolov4
        ├── yolov4.cfg
        └── yolov4.weights
```

Using tkDNN to accelerate yolov4 inferencing required 3 followings steps

* Export weights from darknet

```bash
## a. Download customized darknet used to export weights
cd darknet; make
## b. export weights
mkdir layers debug
./darknet export ../data/yolov4/yolov4.cfg ../data/yolov4/yolov4.weights layers
## c. then move layers, debug directory into this directory for storing
├── data          
    ├── yolov4
        ├── debug
        ├── layers 

```


* Combine into RT file. Replace content of yolo4.cpp main method by following.(it isJust change some path, no big deal). The the modified file is at [yolo4.cpp](https://raw.githubusercontent.com/gachiemchiep/source_code/master/yolov4/tkDNN/yolo4.cpp)

```cpp
int main() {
    std::string bin_path  = "../data/yolov4";
    std::vector<std::string> input_bins = { 
        bin_path + "/layers/input.bin"
    };
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer139_out.bin",
        bin_path + "/debug/layer150_out.bin",
        bin_path + "/debug/layer161_out.bin"
    };
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path  = bin_path + "/yolov4.cfg";
    std::string name_path = bin_path + "/coco.names";
    std::cout << cfg_path << std::endl;
    std::cout << name_path << std::endl;

    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
    net->print();

    //convert network to tensorRT
    tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName("yolov4"));
    
    int ret = testInference(input_bins, output_bins, net, netRT);
    net->releaseLayers();
    delete net;
    delete netRT;
    return ret;
}

```

```bash
## b. create model in tensorRT format (.rt file)
cd tkDNN.build
make

# Export fp32 model
rm yolo4_fp32.rt; 
export TKDNN_MODE=FP32
./test_yolo4
# Export fp16 model
rm yolo4_fp16.rt; 
export TKDNN_MODE=FP16
./test_yolo4
```

* Then test the inferencing

```bash
# fp32
./demo yolov4_fp32.rt ../tkDNN/demo/yolo_test.mp4 y
# fp16
./demo yolov4_fp16.rt ../tkDNN/demo/yolo_test.mp4 y
```

### Result

```bash
# yolov4 608x608 fp32
...
RtBuffer 0   dim: Data dim: 1 3 608 608 1
RtBuffer 1   dim: Data dim: 1 255 76 76 1
RtBuffer 2   dim: Data dim: 1 255 38 38 1
RtBuffer 3   dim: Data dim: 1 255 19 19 1
camera started
....

Time stats:
Min: 46.8783 ms
Max: 88.2218 ms
Avg: 49.6727 ms	20.1318 FPS

# yolov4 608x608 fp16
RtBuffer 0   dim: Data dim: 1 3 608 608 1
RtBuffer 1   dim: Data dim: 1 255 76 76 1
RtBuffer 2   dim: Data dim: 1 255 38 38 1
RtBuffer 3   dim: Data dim: 1 255 19 19 1


Time stats:
Min: 46.6338 ms
Max: 66.4354 ms
Avg: 51.1639 ms	19.545 FPS

```

## OpenCV

### Prepare

We need to compile OpenCV using the following command. If you're using a newer card, please add your architecture to the *CUDA_ARCH_BIN* parameters. See [Matching SM architectures (CUDA arch and CUDA gencode) for various NVIDIA cards](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for a list


```bash
# Download the lastest version of opencv and opencv_contrib and put at following location
~/opt/opencv
├── opencv                  : opencv master
├── opencv_contrib          : opencv contrib master branch
├── opencv.build            : build directory

# Install minoconda at ~/miniconda3 and create the py2 and py3 environment
conda create --name py2 python=2.7
conda create --name py3 python=3.7

# activate thoese environments and install numpy 
conda activate py2; pip install numpy

# Build opencv
cd opencv.build
# Note: check whether your card can support CUDA_ARCH_BIN
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opt/opencv/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D BUILD_EXAMPLES=OFF ..\
	-D BUILD_DOCS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_TESTS=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=ON \
    -D HAVE_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_FORCE_PYTHON_LIBS=ON \
    -D PYTHON2_EXECUTABLE=~/miniconda3/envs/py2/bin/python \
    -D PYTHON2_LIBRARY=~/miniconda3/envs/py2/lib/libpython2.7.so \
    -D PYTHON2_INCLUDE_DIRS=~/miniconda3/envs/py2/include \
    -D PYTHON2_NUMPY_INCLUDE_DIRS=~/miniconda3/envs/py2/lib/python2.7/site-packages/numpy \
    -D PYTHON3_EXECUTABLE=~/miniconda3/envs/py3/bin/python \
    -D PYTHON3_LIBRARY=~/miniconda3/envs/py3/lib/libpython3.7m.so \
    -D PYTHON3_INCLUDE_DIRS=~/miniconda3/envs/py3/include \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=~/miniconda3/envs/py3/lib/python3.7/site-packages/numpy \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_NVCUVID=ON  \
	-D OPENCV_DNN_CUDA=ON \
    -D WITH_FFMPEG=1 \
    -D WITH_TIFF=ON \
    -D WITH_CUBLAS=1 \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
    -D CUDA_ARCH_BIN="6.0 6.2 7.0 7.5" -D CUDA_ARCH_PTX="" ../opencv

make -j8

# Copy into corresponding environments
# py2
cp lib/cv2.so ~/miniconda3/envs/py2/lib/python2.7/site-packages
# py3
cp lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so ~/miniconda3/envs/py3/lib/python3.7/site-packages/
```

Note : Remember to compile with **OPENCV_DNN_CUDA** or else we won't be able to use the **DNN_TARGET_CUDA_FP16**

### Testing with Yolov4 (608x608)

The model *YOLOv4 (608x608): is used to test. Download [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) , [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg), [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names), [demo_wc.py](), [yolo_test.mp4](https://github.com/ceccocats/tkDNN/blob/master/demo/yolo_test.mp4?raw=true). 


```bash
# Then use following scrip to run yolov4 using tkDNN demo video. 
python demo_wc.py --cfg /datadrive/workspace/tkDNN/data/yolov4/yolov4.cfg --weight /datadrive/workspace/tkDNN/data/yolov4/yolov4.weights --shape 608 --batch_size 1 --classes /datadrive/workspace/tkDNN/data/yolov4/coco.names --file_path /datadrive/workspace/tkDNN/tkDNN/demo/yolo_test.mp4
```

### Result

```bash
# fp32
Namespace(batch_size=1, cfg='/datadrive/workspace/tkDNN/data/yolov4/yolov4.cfg', classes='/datadrive/workspace/tkDNN/data/yolov4/coco.names', conf=0.5, file_path='/datadrive/workspace/tkDNN/tkDNN/demo/yolo_test.mp4', mode='fp32', nms=0.6, shape=608, weight='/datadrive/workspace/tkDNN/data/yolov4/yolov4.weights')
Init network in: 689.98 ms
Warmup network in: 482.0 ms
Min: 59.8621 ms
Max: 74.6419 ms
Avg: 61.6272 ms 16.2266 FPS

# fp16 : maybe fp16 didn't work on my GTX1060

Namespace(batch_size=1, cfg='/datadrive/workspace/tkDNN/data/yolov4/yolov4.cfg', classes='/datadrive/workspace/tkDNN/data/yolov4/coco.names', conf=0.5, file_path='/datadrive/workspace/tkDNN/tkDNN/demo/yolo_test.mp4', mode='fp16', nms=0.6, shape=608, weight='/datadrive/workspace/tkDNN/data/yolov4/yolov4.weights')
Init network in: 707.43 ms
Warmup network in: 3075.9 ms
^CMin: 2595.4211 ms
Max: 2636.2209 ms
Avg: 2614.7480 ms 0.3824 FPS

```

### Note

The postprocess which is being used at [https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py) run very slow. In some case postprocessing will cost more time than the inferencing. So we re-write using numpy and matrix manipulation. The result are as follow 


```bash
# Ver 0 : use opencv version
1 loop, best of 3: 27.4 s per loop
-> postprocess = 27.4 / 100 = 0.274 sec = 274 ms

# Ver 1 : use numpy argmax (remove the opencv for loop)
1 loop, best of 3: 1.46 s per loop
-> postprocess = 1.46 / 100 = 0.0146 = 14.6 ms

# Ver 2 : merge features and do postprocess for each image
1 loop, best of 3: 1.54 s per loop
-> postprocess  = 1.54 / 100 = 0.0154 = 15.4 ms

# Ver 3: merge all features, use confidence threshold to remove invalid bbox, then do nms for each image
# note : each image has different valid bboxes. so in the last step we must use the for loop
1 loop, best of 3: 1.45 s per loop
-> post process = 1.45 / 100 = 0.0145 = 14.5 ms

```

## TVM

DIDN'T WORK YET

### Preparation

Follow [tvm from source](https://tvm.apache.org/docs/install/from_source.html#install-from-source) to compile and install tvm python wrapper

```bash
# install llvm : https://apt.llvm.org/
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
# use the config.make at : 
# or edit as https://tvm.apache.org/docs/install/from_source.html

cd build
cmake ..
make -j4

# use the py3 environment
conda activate py3
cd python; python setup.py install;
cd topi/python; python setup.py install;

```

### Testing with Yolov4 (608x608)

The model *YOLOv4 (608x608): is used to test. Download [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) , [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg), [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names), [demo_wc.py](), [yolo_test.mp4](https://github.com/ceccocats/tkDNN/blob/master/demo/yolo_test.mp4?raw=true). 

### Result 

The [Compile YOLO-V2 and YOLO-V3 in DarkNet Models](https://tvm.apache.org/docs/tutorials/frontend/from_darknet.html) haven't work for yolov4 yet

## Result comparing

Device = GTX1060, input size = 608x608

| FP   |      tkDNN  (ms)    |  OpenCV (ms) |
|----------|:-------------:|------:|
| FP32 |  49.6727 | 61.6272 |
| FP16 |    51.1639   |   XXX |

## Personal thinking

* tkDNN : very fast, but the code isn't mature, need to write C++ code
* OpenCV : fast, easy to use, don's need to much step, can use under C++ and Python
* TVM : not available yet, but based on the official document, it can by used under Python

## Reference

1. [tkDNN](https://github.com/ceccocats/tkDNN)
2. [tensorflow install gpu](https://www.tensorflow.org/install/gpu)
3. [Matching SM architectures (CUDA arch and CUDA gencode) for various NVIDIA cards](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
4. [tkDNN install opencv4](https://github.com/ceccocats/tkDNN/blob/master/scripts/install_OpenCV4.sh)
5. [opencv dnn](https://docs.opencv.org/master/df/d57/namespacecv_1_1dnn.html)
6. [tvm](https://github.com/apache/incubator-tvm)
7. [tvm: yolov2 + yolov3](https://tvm.apache.org/docs/tutorials/frontend/from_darknet.html)
8. [tvm install from source](https://tvm.apache.org/docs/install/from_source.html#install-from-source)