
# 6DOF = 6 dimension of freedom = position (x, y, z) + orientation (roll, pitch, yawn)

Read those articles to understand about the pose estimatin

## [OpenCV Pose Estimation](https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html)

Simple explaination

## [Pose estimation with ROS and ArUco marker](https://github.com/XuanliangDeng/ROS-ArUco-marker-detection-and-6DOF-pose-estimation-of-USB-camera)

First he does the camera calibration to get the following information *D, K, R, P* from camera. These information can be extracted by reading from Ros's corresponding *CameraInfo* topic

```bash
# The distortion parameters, size depending on the distortion model.
# For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
# float64[] D


# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
# Rectification matrix (stereo cameras only)
# A rotation matrix aligning the camera coordinate system to the ideal
# stereo image plane so that epipolar lines in both stereo images are
# parallel.
# float64[9]  R # 3x3 row-major matrix

# Projection/camera matrix
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
```

Then he run the [aruco_ros](https://github.com/pal-robotics/aruco_ros)
-> quite the same thing



## [DOPE-ROS-D435](https://github.com/yehengchen/DOPE-ROS-D435)

-> the most simple ROS 6DOF