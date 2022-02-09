#!/bin/bash
cd
gnome-terminal -e "bash -c \"source Velero.sh; exec bash\""
gnome-terminal -e "bash -c \"source Puerto_serie.sh; exec bash\""
source Anaconda.sh
cd Documentos/Ubuntu_master/SNN_Codes/Spiking_codes/
python Experiments.py
