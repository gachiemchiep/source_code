# Yolov4 Related

Re-produce the inference speed of Yolov4 reported at https://github.com/AlexeyAB/darknet for the following framework


- [ ] tkDNN
- [x] OpenCV
- [] TVM

## tkDNN

https://github.com/ceccocats/tkDNN#how-to-compile-this-repo

## OpenCV

We need to compile OpenCV using the following command

```bash
# Download the lastest version of opencv and opencv_contrib and put at following location

# 

# get newest opencv 4 (master) (yolov4 include) and compile
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opt/opencv/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_VTK=ON \
    -D BUILD_EXAMPLES=ON ..\
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=ON \
    -D HAVE_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_FORCE_PYTHON_LIBS=ON \
    -D PYTHON2_EXECUTABLE=~/miniconda3/envs/py2/bin/python \
    -D PYTHON2_LIBRARY=~/miniconda3/envs/py2/lib/libpython2.7.so \
    -D PYTHON2_INCLUDE_DIRS=~/miniconda3/envs/py2/include \
    -D PYTHON2_NUMPY_INCLUDE_DIRS=~/miniconda3/envs/py2/lib/python2.7/site-packages/numpy \
    -D PYTHON3_EXECUTABLE=~/miniconda3/envs/py3/bin/python \
    -D PYTHON3_LIBRARY=~/miniconda3/envs/py3/lib/libpython3.7m.so \
    -D PYTHON3_INCLUDE_DIRS=~/miniconda3/envs/py3/include \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=~/miniconda3/envs/py3/lib/python3.7/site-packages/numpy \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TIFF=ON \
	-D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=ON \
	-D WITH_LIBV4L=ON \
	-D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
    -D CUDA_ARCH_BIN="6.0 6.1 7.0 7.5" -D CUDA_ARCH_PTX="" \
    -D BUILD_EXAMPLES=ON ../opencv


cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opt/opencv/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_VTK=ON \
    -D BUILD_EXAMPLES=ON ..\
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=ON \
    -D HAVE_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_FORCE_PYTHON_LIBS=ON \
    -D PYTHON2_EXECUTABLE=~/opt/miniconda2/bin/python \
    -D PYTHON2_LIBRARY=~/opt/miniconda2/lib/libpython2.7.so \
    -D PYTHON2_INCLUDE_DIRS=~/opt/miniconda2/include \
    -D PYTHON2_NUMPY_INCLUDE_DIRS=~/opt/miniconda2/lib/python2.7/site-packages/numpy \
    -D PYTHON3_EXECUTABLE=~/opt/miniconda2/envs/py3.6/bin/python \
    -D PYTHON3_LIBRARY=~/opt/miniconda2/envs/py3.6/lib/libpython3.6m.so \
    -D PYTHON3_INCLUDE_DIRS=~/opt/miniconda2/envs/py3.6/include \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=~/opt/miniconda2/envs/py3.6/lib/python3.6/site-packages/numpy \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TIFF=ON \
	-D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=ON \
	-D WITH_LIBV4L=ON \
	-D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
    -D CUDA_ARCH_BIN="6.0 6.1 7.0 7.5" -D CUDA_ARCH_PTX="" \
    -D BUILD_EXAMPLES=ON ../opencv
```

Remember to compile with **OPENCV_DNN_CUDA** or else we won't be able to use the **DNN_TARGET_CUDA_FP16**

The postprocess which is being used at [https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py) run very slow. In some case postprocessing will cost more time than the inferencing. So we re-write using numpy and matrix manipulation. The result are as follow 


```bash
# Ver 0 : use opencv version
1 loop, best of 3: 27.4 s per loop
-> postprocess = 27.4 / 100 = 0.274 sec = 274 ms
	-> 91 ms per image

# Ver 1 : use numpy argmax (remove the opencv for loop)
1 loop, best of 3: 1.46 s per loop
-> postprocess = 1.46 / 100 = 0.0146 = 14.6 ms
	-> 5ms per image

# Ver 2 : merge features and do postprocess for each image
1 loop, best of 3: 1.54 s per loop
-> postprocess  = 1.54 / 100 = 0.0154 = 15.4 ms
	-> 5ms per image

# Ver 3: merge all features, use confidence threshold to remove invalid bbox, then do nms for each image
# note : each image has different valid bboxes. so in the last step we must use the for loop
1 loop, best of 3: 1.45 s per loop
-> post process = 1.45 / 100 = 0.0145 = 14.5 ms

-> 1, 2, 3 is very close
-> Ver 3 is fast but very hard to mainternance the code . 
So we will use version 2 instead

```

## TVM