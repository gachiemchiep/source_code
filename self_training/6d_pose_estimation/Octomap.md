

## ...

voxel = 3d pixel

voxel octrees : [Youtube : Voxel Octrees: What Are They?](https://www.youtube.com/watch?v=mcpLSHU8M1c)

octomap explaination : [ros blog](http://ros-developer.com/2017/11/27/octomap-explanierend/)

## Tutorials

### 1. [Making occupancy grid map in ROS from gazebo with Octomap](http://ros-developer.com/2017/05/02/making-occupancy-grid-map-in-ros-from-gazebo-with-octomap/)

```bash
conda install python=2
conda install -c conda-forge xorg-libsm

# Installation
sudo apt install ros-melodic-gmapping
sudo apt install ros-melodic-turtlebot*
# See this link to run turtlebot-3
# https://cyaninfinite.com/installing-turtlebot-3-simulator-in-ubuntu-14-04/

# Download the following ros package
octomap
octomap_mapping
octomap_ros

# launch gazebo world
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_gazebo turtlebot3_world.launch

# rviz
roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch

# keyboard
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

# octomap 
 roslaunch octomap_turtlebot.launch 

# inside rviz show the following topic
/occupied_cells_vis_array
```

### 2. [How to create octomap tree from mesh](http://ros-developer.com/tag/octomap/)

-> not useful, next




## Reference 
1. [Making occupancy grid map in ROS from gazebo with Octomap](http://ros-developer.com/2017/05/02/making-occupancy-grid-map-in-ros-from-gazebo-with-octomap/)


2. [How to create octomap tree from mesh](http://ros-developer.com/tag/octomap/)
