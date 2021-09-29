#from multiprocessing import Process
import Simulation as sp
import SNN_network as SNN
import environment as env

#import torch
#import matplotlib.animation as animation

# Serial port
port_name='interface_1'
direction= '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes'
timeout=15  
p = sp.serial_port(direction, port_name, timeout)
# Sensor ranges

vmax=[500, 180, 30, 30, 180]
vmin=[0, -180, -30, -30, -180]
AM=[20,90]
Am=[-20,-90]
# Size step of algorithm
PMax = 1
Pmin = 0.01
step = 5e-1
# Simulation and control times
factor_time = 1
simul_time = 3
reward_time = simul_time/factor_time
time_network=250
# Network characteristics and connections
dt=1
control_signals=4
number_actuators=1
min_freq=0
max_freq=1
codify = 'bernoulli'
#dato=[1,1,0,1,1,1]
#dato=[0.9, 1, 0.6, 0.8, 0.3]
#dato=[0.8,0.51,0.33,0.22,0.1]
neur=[control_signals*2, 15,number_actuators]

i= input("Desea cargar la red  (cero) o crear una nueva (uno)?")

if i=='1':

    max_spikes,min_spikes = SNN.max_min_spikes(neuronas=neur,
                                dt=dt,
                                time=time_network,
                                recurrent=0,
                                modelo='LIF',
                                wM = PMax,
                                wm = Pmin,
                                n = step,
                                codify = codify,         
                                maximum = max_freq,
                                minimun = min_freq
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


else:
    
    [capas,monitores,net, m_pesos, max_spikes, min_spikes, neur] = SNN.cargar_red(name='red_prueba.pt', 
                                                                                  direction=direction, 
                                                                                  learning = True)
    
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
restart=2
state = 0
saving = 0
# band=p.open_port()
# while True:
#     p.classic_controller()

while True:
    if not band:
        band=p.open_port()
    else:
        #p.classic_controller()
        dato=p.read_data_sensor()
        if not isinstance(dato,bool):
            [n_dato,re] = env.control_inputs(dato,vmax,vmin, state, max_freq, min_freq);
            rew[1] = re
            if restart==2:
                state = 0
            if restart ==2 or restart ==1:
                ro,r = env.get_plane(dato)           
                env.carril_velero(ro,r,2)
                r=0
                net.reset_state_variables()
            else:        
                r = env.reward(rew,1,limit = 1) #recompensa tipo 1
                #r = env.reward(dato,2,limit = 1)
            n_dato = n_dato+list(max_freq*env.np.ones(len(n_dato)))
            print(n_dato[])
            #cod_n_dato = SNN.codificacion_red('bernoulli',1,dt,n_dato)
            #net.run(inputs={capas[0] : cod_n_dato}, time=1, reward=-1)
            cod_n_dato = SNN.codificacion_red(codify,time_network,dt,n_dato)
            net.run(inputs={capas[0] : cod_n_dato}, time=time_network, reward=r)
            control_action=SNN.decode_spikes(monitores[len(neur)-1],[max_spikes,min_spikes],AM,Am,rew[1])
            control_action[2] = env.is_restart([dato[1],dato[2]], control_action[2])
            p.write_control_action(control_action)
            SNN.imprimir_spikes(monitores,time_network,s1,s2)
            SNN.imprimir_pesos(m_pesos, wmax=PMax, wmin=Pmin, im=im)
            if(control_action[2]==1):
                state += 1
                if saving < state:
                    SNN.guardar_red('red_prueba.pt',net,direction,max_spikes,min_spikes,neur)
                    saving +=1
            
            rew[0]=rew[1]
            restart = control_action[2]
        else:
            print("No hay dato")

# ztraining = Process(target=SNN_training)
# graph = Process(target=Graphs_generator,args=(False,))
# monitores = []
# 
# graph.start()
# training.start()
