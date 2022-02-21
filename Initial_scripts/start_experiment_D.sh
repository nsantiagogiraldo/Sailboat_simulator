#!/bin/bash
cd /home/nelson
chmod +x Puerto_serie.sh
chmod +x Anaconda.sh
chmod +x Velero.sh
./ Puerto_serie.sh &
./ Velero.sh &
source Anaconda.sh
cd Documentos/Ubuntu_master/SNN_Codes/Spiking_codes/
python Experiments.py
