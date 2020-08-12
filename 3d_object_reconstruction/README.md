#  Part Reconstruction

- [ ] Next best view
- [ ] Reconstruction algorithm
- [ ] Other next best view implementation

## Detail 

### next_best_view

* surfel_next_best_view

    2014: [https://github.com/RMonica/basic_next_best_view](https://github.com/RMonica/basic_next_best_view)
        -> [paper link](http://www.ce.unipr.it/~rizzini/papers/aleotti14iros.pdf)

    2018 : https://github.com/RMonica/surfel_next_best_view

    2020 : https://github.com/RMonica/surfels_unknown_space

    Lab's link : http://www.rimlab.ce.unipr.it/Software.html

    -> license : private license, weak, no need money  

    -> couldn't find the paper

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



### Global Registration of Mid-Range 3D Observations and Short Range Next Best Views

* [source code](https://github.com/RMonica/basic_next_best_view)
* [paper link](http://www.ce.unipr.it/~rizzini/papers/aleotti14iros.pdf)

* Detail summary 

```bash

```

## Reference

1. [ROS blog: Intelligent Part Reconstruction](https://rosindustrial.org/news/2018/1/3/intelligent-part-reconstruction)
2. [Python robotics: not useful](https://github.com/AtsushiSakai/PythonRobotics)