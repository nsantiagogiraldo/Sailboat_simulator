#!/bin/bash
source /opt/ros/kinetic/setup.bash
source /home/nelson/catkin_ws/install_isolated/setup.bash
source /home/nelson/Documentos/Ubuntu_master/Initial_scripts/Velero.sh &
sleep 20
rosservice call gazebo/unpause_physics

