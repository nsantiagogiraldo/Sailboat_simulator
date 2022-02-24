#!/bin/bash
cd /home/nelson
mkdir -p Documentos/Ubuntu_master
cd Documentos/Ubuntu_master
git clone https://github.com/nsantiagogiraldo/Sailboat_simulator.git

rm  /home/nelson/catkin_ws/src/usv_sim_lsa/usv_base_ctrl/scripts/sailboat_control_heading.py
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_dynamics/foil_dynamics_plugin/src/foil_dynamics_plugin.cpp
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_dynamics/foil_dynamics_plugin/include/foil_dynamics_plugin/foil_dynamics_plugin.h
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_navigation/scripts/patrol_pid_scene2.py
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/launch/scenarios_launchs/sailboat_scenario2.launch
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/scenes/sailboat_scenario2.xml
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/world/empty_accelerated.world
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/xacro/sailboat.xacro 
rm /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/xacro/boat_subdivided4.xacro

cp  /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/boat_subdivided4.xacro /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/xacro
cp  /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/sailboat.xacro /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/xacro
cp  /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/communicate.py /home/nelson/catkin_ws/src/usv_sim_lsa/usv_base_ctrl/scripts
cp  /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/sailboat_control_heading.py /home/nelson/catkin_ws/src/usv_sim_lsa/usv_base_ctrl/scripts
cp  /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/text_file.py /home/nelson/catkin_ws/src/usv_sim_lsa/usv_base_ctrl/scripts
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/foil_dynamics_plugin.cpp /home/nelson/catkin_ws/src/usv_sim_lsa/usv_dynamics/foil_dynamics_plugin/src
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/foil_dynamics_plugin.h /home/nelson/catkin_ws/src/usv_sim_lsa/usv_dynamics/foil_dynamics_plugin/include/foil_dynamics_plugin
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/patrol_pid_scene2.py /home/nelson/catkin_ws/src/usv_sim_lsa/usv_navigation/scripts
cp cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/sailboat_scenario2.launch /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/launch/scenarios_launchs
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/sailboat_scenario2.xml /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/scenes
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/empty_accelerated.world /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/world
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/sailboat.xacro /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/xacro
cp /home/nelson/Documentos/Ubuntu_master/Versions/Actual_version/boat_subdivided4.xacro /home/nelson/catkin_ws/src/usv_sim_lsa/usv_sim/xacro

cp /home/nelson/Documentos/Ubuntu_master/Initial_scripts/start_experiment_D.sh /home/nelson/start_experiment.sh
cp /home/nelson/Documentos/Ubuntu_master/Initial_scripts/Velero_D.sh /home/nelson/Velero.sh
cp /home/nelson/Documentos/Ubuntu_master/Initial_scripts/Puerto_serie.sh /home/nelson
cp /home/nelson/Documentos/Ubuntu_master/Initial_scripts/Anaconda_D.sh /home/nelson/Anaconda.sh



