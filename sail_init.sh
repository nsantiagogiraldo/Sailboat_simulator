#!/bin/bash
DIR="$( cd "$( dirname "$0" )" && pwd )"
source /opt/ros/kinetic/setup.bash
source ~/catkin_ws/install_isolated/setup.bash
source $DIR/sailboat.sh &
sleep 20
rosservice call gazebo/unpause_physics

