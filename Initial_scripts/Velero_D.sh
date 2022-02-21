#!/bin/bash
source /opt/ros/kinetic/setup.bash
cd "catkin_ws"
source install_isolated/setup.bash
roslaunch usv_sim sailboat_scenario2.launch parse:=true
roslaunch usv_sim sailboat_scenario2.launch parse:=false

