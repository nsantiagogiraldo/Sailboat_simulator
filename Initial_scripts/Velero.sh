#!/bin/bash
source /opt/ros/kinetic/setup.bash
source /home/nelson/catkin_ws/install_isolated/setup.bash
cd /home/nelson/catkin_ws && roslaunch usv_sim sailboat_scenario2.launch parse:=true && roslaunch usv_sim sailboat_scenario2.launch parse:=false

