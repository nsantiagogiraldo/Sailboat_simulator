#!/bin/bash
#Execute as root access
cd /home/nsantiago
paralell=$((6))
# Create a paralell containers. 
for (( c=1; c<=$paralell; c++ ))
do  
  name[$c]="My"$c
  docker create --name ${name[$c]} --memory-swap -1 -it  nsantiagogiraldo/usvsim_ubuntu16.04 
done
exp=$(ls config | wc -l)
cont=$((0))
# Create if path does not exist
mkdir -p results/data
mkdir -p results/controllers
# Obtain experiments to simulate
echo $exp > trained.txt
ls results/data >> trained.txt
python config.py
sleep 5

text="config"
txt=".txt"
while IFS= read -r line
do
  conf[$cont]=$text$(($line))$txt
  cont=$(($cont+1))
done < trained.txt
rm trained.txt
# Launch all the experiments

used=((0))
var=$paralell
until (( ${#conf[@]}==0 && $var==$paralell))
do  
  if (( ${#conf[@]}!=0 ))
  then
    sleep 60
  else
    sleep 600
  fi
  p=$(docker ps --filter "status=created" --filter "status=exited" --format "{{.Names}}")
  used=($p)
  var=${#used[@]}
  while (( ${#used[@]}>0 ))
  do
    docker cp ${used[0]}:/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes/data_result/. /home/nsantiago/results/data
    docker cp ${used[0]}:/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes/control/. /home/nsantiago/results/controllers
    if (( ${#conf[@]}!=0 ))
    then
      docker cp config/${conf[0]} ${used[0]}:/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes/config.txt
      unset conf[0]
      conf=(${conf[*]})
      docker start ${used[0]}
    fi
    unset used[0]
    used=(${used[*]})
  done
done

# Delete containers
for (( c=1; c<=$paralell; c++ ))
do  
  docker rm ${name[$c]}
done




