# USVSIM sailboat simulator

## Simulator installation:

1. Install USVSIM simulator, full instructions and requirements are on the page https://github.com/disaster-robotics-proalertas/usv_sim_lsa.

2. Before running the catkin_make_isolated --install line, copy the hidden .uwsim folder (found at the address catkin_ws / src / usv_sim_lsa) to your home folder and to the uwsim_resources folder (found at the address catkin_ws / src / usv_sim_lsa). The UWSIM package may take a long time to install, so please be patient.

3. Copy the Sailboat.sh script, from the Initial_scripts folder, in your personal folder.

4. Run the Sailboat.sh script and immediately run the following two lines: roslaunch usv_sim sailboat_scenariox.launch parse: = true, roslaunch usv_sim sailboat_scenariox.launch parse: = false, where x is the scenario you want to run. This will open the default simulation of scenario x with the generic USVSIM sailboat.

## Modified USVSIM sailboat simulator.

To view the modified sailboat in its first version, perform the following steps:

1. Replace the boat_subdivided4.xacro and sailboat.xacro files in the catkin_ws / src / usv_sim_lsa folder with the files of the same name in the Versions / Actual_version folder.

2. Open a terminal and run the Sailboat.sh script

3. Run the command catkin_make_isolated --install

4. Execute the two normal opening commands for a simulation scenario (point 4 of the installation)

### If you want to modify the model of the sailboat:

1. Modify the sailboat.xacro, boat_subdivided4.xacro, sail.dae and box.dae files found inside the catkin_ws / src / usv_sim_lsa folder

2. Execute all the steps from point 2 of the previous section.

### If you want to modify a scenario:

1. Modify the files patrol_pid_scenex.py and sailboat_scenariox.xml, where x is the number of the scenario you want to modify

2. Execute all the steps from point 2 of the previous section.

### If you want to modify the sailboat controller:

1. Modify the sailboat_control_heading.py file

2. Execute all the steps from point 2 of the previous section

