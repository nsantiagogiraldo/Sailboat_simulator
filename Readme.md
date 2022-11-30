# SNN-based sailing control systems simulator (In process of documentation)

This simulation environment allows training spiking neural networks (SNNs) to perform navigation tasks for sailboats. Initially, we used it to develop a SNN-based navigation control system to reach a target point, but it could be adapted to other applications. This simulation environment is based on the [USVSim simulator](https://github.com/disaster-robotics-proalertas/usv_sim_lsa) and the [BindsNET library](https://bindsnet-docs.readthedocs.io/). Additionally, we use [Socat](http://www.dest-unreach.org/socat/) to establish a communication link between USVSim and BindsNET. USVSim is used to simulate the sailboat and its physical environment (waves, water currents and wind), and BindsNET is used to implement the SNNs of our control system. This simulator can be used to test and improve our control strategy, as well as to explore how to use SNN in other sailboat control applications.


## Prerequisites

Our simulation environment requires you to install the USVSim environment, the BindsNET library and Socat. Due to USVSim installation conditions, this environment works only on ubuntu 16.04.

We recommend installing the BindsNET library using the Anaconda environment (python 3.8), with the following commands:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch 
pip install git+https://github.com/BindsNET/bindsnet.git
```

## Installing



```bash
git clone https://github.com/nsantiagogiraldo/Sailboat_simulator.git

```
