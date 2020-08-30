#  6d pose estimation

## Detail 

1. [3d reconstruction](3d_reconstruction.md)
2. [data prepation]



# course

tsdf : http://graphics.cs.cmu.edu/courses/15769/fall2016content/lectures/16_realtime3d/16_realtime3d_slides.pdf

Computer Graphics (CMU 15-462/662)
http://15462.courses.cs.cmu.edu/fall2019/



## Other


* Very interesting project 

    https://github.com/felixchenfy/Detect-Object-and-6D-Pose

    [1. Scann object's 3D model](https://github.com/felixchenfy/3D-Scanner-by-Baxter)
        -> oop the robot move by pre-defined location
        -> we want to generate the position automatically 


    [2. Collect data for training Yolo](https://github.com/felixchenfy/Mask-Objects-from-RGBD)

    [3. Train Yolo](https://github.com/felixchenfy/Data-Augment-and-Train-Yolo)

    [4. Locate object 6D pose] TODO 

-> all those respository are forked.

* [https://github.com/zjudmd1015/Mini-3D-Scanner](https://github.com/zjudmd1015/Mini-3D-Scanner)

    -> put the object on rotation disk, then scan and merge the point cloud. But the yak already do this very well 

* [http://wiki.ros.org/next_best_view](http://wiki.ros.org/next_best_view)  : last update in 2010, very old

* [asr_next_best_view](https://github.com/asr-ros/asr_next_best_view)

* [Receding Horizon Next Best View Planning](https://github.com/ethz-asl/nbvplanner) : https://github.com/ethz-asl/nbvplanner/wiki/Demo-Scenario <- use for uav

* [Next Best View Exploration](https://github.com/kucars/nbv_exploration) : environment 3d explorsion

* [https://github.com/protontypes/awesome-robotic-tooling](https://github.com/protontypes/awesome-robotic-tooling) : <- not useful

-> what we can do to improve the above project

- [ ] use yak to make better model
- [ ] replace Baster robot by UR5
- [ ] Apply next best view planner
- [ ] 6D pose estimation 
- [ ] 2d for tracking, 3d for pose estimation
- [ ] gripper



## Reference

1. [ROS blog: Intelligent Part Reconstruction](https://rosindustrial.org/news/2018/1/3/intelligent-part-reconstruction)
2. [Python robotics: not useful](https://github.com/AtsushiSakai/PythonRobotics)