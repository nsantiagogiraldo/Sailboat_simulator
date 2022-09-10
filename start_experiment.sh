#!/bin/bash
DIR="$( cd "$( dirname "$0" )" && pwd )"
gnome-terminal -e "bash -c \"cd $DIR; source sail_init.sh; exec bash\""
gnome-terminal -e "bash -c \"cd $DIR;source Puerto_serie.sh; exec bash\""
source $DIR/Anaconda.sh
sleep 20
python $DIR/SNN_Codes/Spiking_codes/Experiments.py
pkill terminal
