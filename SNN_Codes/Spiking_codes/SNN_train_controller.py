#from multiprocessing import Process
import Simulation as sp
import SNN_network as SNN
#import torch
#import matplotlib.animation as animation

# Serial port
port_name='interface_1'
direction= '/home/nelson/Documentos/Ubuntu_Maestria/SNN_Codes/Canal-Puertos'
timeout=15  
p = sp.serial_port(direction, port_name, timeout)
# Sensor ranges

vmax=[500, 180, 45, 45, 180]
vmin=[0, -180, -45, -45, -180]
AM=[20,90]
Am=[-20,-90]
# Size step of algorithm
PMax = 1
Pmin = 0.01
step = 1e-1
# Simulation and control times
factor_time = 1
simul_time = 3
reward_time = simul_time/factor_time
time_network=500
# Network characteristics and connections
dt=1
control_signals=4
number_actuators=1
#dato=[1,1,0,1,1,1]
#dato=[0.9, 1, 0.6, 0.8, 0.3]
#dato=[0.8,0.51,0.33,0.22,0.1]
neur=[control_signals*2, number_actuators]

max_spikes,min_spikes = SNN.max_min_spikes(neuronas=neur,
                            dt=dt,
                            time=time_network,
                            recurrent=0,
                            modelo='LIF',
                            wM = PMax,
                            wm = Pmin,
                            n = step
                            )

[capas,monitores,net, m_pesos] = SNN.arquitectura_red(neuronas=neur,
                                          dt=dt,
                                          time=time_network,
                                          recurrent=0,
                                          modelo='LIF',
                                          wM = PMax,
                                          wm = Pmin,
                                          n = step
                                          )
#print(max_spikes)
#print(min_spikes)
# data = SNN.codificacion_red('bernoulli',time_network,dt,dato)
# net.run(inputs={capas[0] : data}, time=time_network,reward=0)
# uu=SNN.decode_spikes(monitores[1],max_spikes,AM,Am,10)
#SNN.imprimir_spikes(monitores,time_network,None,None)

#im = SNN.imprimir_pesos(m_pesos,10,0,None)
#print(uu)

#SNN controller

band=False
rew = [0,0]
s1=None
s2=None
im=None
while True:
    if not band:
        band=p.open_port()
    else:
        #p.classic_controller()
        dato=p.read_data_sensor()
        if not isinstance(dato,bool):
            [n_dato,re] = SNN.control_inputs(dato,vmax,vmin);
            rew[1] = re
            r = SNN.reward(rew,1,limit = 2)
            n_dato = n_dato+list(SNN.np.ones(len(n_dato)))
            cod_n_dato = SNN.codificacion_red('bernoulli',1,dt,n_dato)
            net.learning=True
            net.run(inputs={capas[0] : cod_n_dato}, time=1, reward=r)
            cod_n_dato = SNN.codificacion_red('bernoulli',time_network,dt,n_dato)
            net.learning=False
            net.run(inputs={capas[0] : cod_n_dato}, time=time_network, reward=-1)
            control_action=SNN.decode_spikes(monitores[len(neur)-1],[max_spikes,min_spikes],AM,Am,rew[1])
            p.write_control_action(control_action)
            SNN.imprimir_spikes(monitores,time_network,s1,s2)
            SNN.imprimir_pesos(m_pesos, wmax=PMax, wmin=Pmin, im=im)
            print(r)
            print(control_action)
            rew[0]=rew[1]
        else:
            print("No hay dato")

# ztraining = Process(target=SNN_training)
# graph = Process(target=Graphs_generator,args=(False,))
# monitores = []
# 
# graph.start()
# training.start()
