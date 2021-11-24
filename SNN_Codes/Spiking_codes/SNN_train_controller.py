import Simulation as sp
import SNN_network as SNN
import original_controller as ctr
import environment as env

Learning = True

# Serial port
port_name='interface_1'
direction = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes'
timeout=15  
p = sp.serial_port(direction, port_name, timeout)

# Sensor ranges
vmax=[60, 30, 30, 180]
vmin=[-60, -30, -30, -180]
AM=[30,90]
Am=[-30,-90]
# Size step of algorithm
PMax = 10
Pmin = 0
step = 1e-0
# Simulation and control times
factor_time = 1
simul_time = 3
reward_time = simul_time/factor_time
time_network=500
# Network characteristics and connections
hyperparam = [12,1,180,9,90,45,6]  #K1,K2,tacking area, states number, no go zone, tacking angle, channel length
files_names = ['rudder1', 'sail1']
dt=1
control_signals=2*hyperparam[0]*hyperparam[1]
number_actuators=1
min_freq=0
max_freq=240
codify = 'poisson'
redundance = 2
neur=[control_signals*redundance,number_actuators]

rudder_ctrl = SNN.spiking_neuron(controller='rudder', 
                                 name = files_names[0], 
                                 time_network = time_network, 
                                 redundance = redundance,
                                 max_out = AM[0], 
                                 min_out = Am [0])

sails_ctrl = SNN.spiking_neuron(controller='sail', 
                                name = files_names[1], 
                                time_network = time_network, 
                                redundance = redundance,
                                max_out = AM[1], 
                                min_out = Am [1])

i= input("Load SNN (0) or new SNN (1)")

if i=='1':
    
    rudder_ctrl.SNN_setup(neurons=neur, dt=dt, time=time_network, recurrent=0, model='LIF',
                          wM = PMax, wm = Pmin, n = step, codify = codify, maximum = max_freq,
                          minimun = min_freq)
    
    sails_ctrl.SNN_setup(neurons=neur, dt=dt, time=time_network, recurrent=0, model='LIF',
                         wM = PMax, wm = Pmin, n = step, codify = codify, maximum = max_freq,
                         minimun = min_freq)
else:
    
    rudder_ctrl.load_SNN(path=direction,learning = True)
    sails_ctrl.load_SNN(path=direction,learning = True)
    
sail_env = env.sailboat_environment(rudder_ctrl = rudder_ctrl, sail_ctrl = sails_ctrl,
                                    vmax = vmax, vmin = vmin, hyperparam = hyperparam, 
                                    path = direction)

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
        data=p.read_data_sensor()
        if not isinstance(data,bool):
            control_action = sail_env.environment_step(data = data, max_rate = max_freq,
                                                       min_rate = min_freq)
            control_action[1] = ctr.sail_ctrl(10,False)
            print(control_action)
            p.write_control_action(control_action)
            # s1,s2 = SNN.imprimir_spikes(monitores,time_network,s1,s2)
            # im = SNN.imprimir_pesos(m_pesos, wmax=PMax, wmin=Pmin, im=im)
            
            rudder_ctrl.print_weigths(im=im)
            sail_env.save_SNN_state() 
        else:
            print("No hay dato")