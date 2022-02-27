#!/bin/bash
source /opt/ros/kinetic/setup.bash
source /home/nelson/catkin_ws/install_isolated/setup.bash
source /home/nelson/Velero.sh &
sleep 20
rosservice call gazebo/unpause_physics

