import Simulation as sp
import SNN_network as SNN
import environment as env

Inference = False

# Serial port
port_name='interface_1'
direction = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes'
timeout=15  
p = sp.serial_port(direction, port_name, timeout)

# Sensor ranges
vmax=[90, 30, 30, 180]
vmin=[-90, -30, -30, -180]
AM=[25,90]
Am=[-25,-90]
# Size step of algorithm
PMax = 10
Pmin = 0
step = 1e-0
# Simulation and control times
time_network=500
# Network characteristics and connections
hyperparam = [12,1,180,7,30,
              60,4,30,9,275,
              100,10,0.42,45]  #K1,K2,tacking area 1, exit states number, tacking area 2, tacking angle, channel length, K3, exit states sail, xcenter_train, ycenter_train, number_of_training_points, max_speed_tacking,tacking angle 2
files_names = ['rudder1', 'sail1']
dt=1
control_signals=[hyperparam[0]*hyperparam[1],3*hyperparam[7]]
number_actuators=1
min_freq=0
max_freq=240
codify = 'poisson'
redundance = [2,2]
recurrent = [0,0]
neur = [[3*control_signals[0]*redundance[0],number_actuators],
       [control_signals[1]*redundance[1],number_actuators]]

if not Inference:

    rudder_ctrl = SNN.spiking_neuron(controller='rudder', 
                                     name = files_names[0], 
                                     time_network = time_network, 
                                     redundance = redundance[0],
                                     max_out = AM[0], 
                                     min_out = Am [0],
                                     exit_state = hyperparam[3])
    
    sails_ctrl = SNN.spiking_neuron(controller='sail', 
                                    name = files_names[1], 
                                    time_network = time_network, 
                                    redundance = redundance[1],
                                    max_out = AM[1], 
                                    min_out = Am [1],
                                    exit_state = hyperparam[8])
    
    i= input("Load SNN (0) or new SNN (1)")
    
    if i=='1':
        
        rudder_ctrl.SNN_setup(neurons=neur[0], dt=dt, time=time_network, recurrent=recurrent[0], 
                              model='LIF', wM = PMax, wm = Pmin, n = step, codify = codify, 
                              maximum = max_freq, minimun = min_freq)
        
        sails_ctrl.SNN_setup(neurons=neur[1], dt=dt, time=time_network, recurrent=recurrent[1], 
                             model='LIF', wM = PMax, wm = Pmin, n = step, codify = codify, 
                             maximum = max_freq, minimun = min_freq)
    else:
        
        rudder_ctrl.load_SNN(path=direction,learning = True)
        sails_ctrl.load_SNN(path=direction,learning = True)
        
    sail_env = env.sailboat_environment(rudder_ctrl = rudder_ctrl, sail_ctrl = sails_ctrl,
                                        vmax = vmax, vmin = vmin, hyperparam = hyperparam, 
                                        path = direction)
    
    #SNN controller
    band=False
    
    while True:
        if not band:
            band=p.open_port()
        else:
            data=p.read_data_sensor()
            if not isinstance(data,bool):
                control_action = sail_env.environment_step(data = data, max_rate = max_freq,
                                                           min_rate = min_freq)
                p.write_control_action(control_action)            
                #rudder_ctrl.print_weigths(im=None)
                #sail_env.save_SNN_state()
            else:
                print("No data")