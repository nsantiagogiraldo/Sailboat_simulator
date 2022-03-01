#!/bin/bash
cd /home/nsantiago
paralell=$((9))
exp=$(ls config | wc -l)
cont=$((0))

mkdir -p results/data
mkdir -p results/controllers
echo $exp > trained.txt
ls results/data >> trained.txt
python config.py
sleep 5

while IFS= read -r line
do
  conf[$cont]=$(($line))
  cont=$(($cont+1))
done < trained.txt

for (( c=1; c<=$paralell; c++ ))
do  
  name="My"$c
  sudo docker run --name $name --memory-swap -1 -d -it  nsantiagogiraldo/usvsim_ubuntu16.04 
  sudo docker stop $name
done



sudo docker stop My15
sudo docker cp config.txt My15:/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes
sudo docker start My15
sudo docker cp My11:/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes/data_result /home/nsantiago/results/data




