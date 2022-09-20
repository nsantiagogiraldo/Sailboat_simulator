import Simulation as sp
import SNN_network as SNN
import environment as env


class SNN_complete_train_test:
    # Last values in config
    PI_test = 0
    test = 0 
    ######################
    # Serial port
    port_name='ttyS3'
    #direction = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes'
    direction = '/home/khadas/Documents/Ubuntu_master/SNN_Codes/Spiking_codes'
    timeout=15
    #p = sp.serial_port(direction, port_name, timeout)
    p = sp.serial_port('/dev', port_name, timeout)
    config_file = 'config.txt'
    
    # Error ranges for each variable[0,1] and pitch range
    vmax=[]
    vmin=[]
    ###################################################
    # Maximum and minimum output actuator signals
    AM=[]
    Am=[]
    ##################################################
    
    # Network characteristics and connections
    hyperparam = [#K1,K2,tacking area 1, exit states number, tacking area 2
                  #tacking angle, channel length, K3, exit states sail, test_distance
                  #ycenter_train, number_of_test_points, max_speed_tacking,tacking angle 2
                  ]  # train_points_1, train_points_2, train_points_3
    files_names = [] 
    redundance = []
    recurrent = []
    
    number_actuators=0
    min_freq=0
    max_freq=0
    codify = ''
    
    # Size step of algorithm
    PMax = 0
    Pmin = 0
    step = 0
    time_network = 0
    dt = 0
    
    control_signals=[]
    neur = []
    sail_env = {}
    
    def __init__(self):
        try:            
            f = open(self.direction+'/'+self.config_file, 'r')
            info = f.read().split('\n')     
            self.permutation = int(info[19])
            for j in range(8):
                for i in info[j].split(','):
                    if j==0:
                        self.vmax.append(int(i))
                    elif j==1:
                        self.vmin.append(int(i))
                    elif j==2:
                        self.AM.append(int(i))
                    elif j==3:
                        self.Am.append(int(i))
                    elif j==4:
                        if i.find('.')!=-1:
                            self.hyperparam.append(float(i))
                        else:
                            self.hyperparam.append(int(i))
                    elif j==5:
                        self.files_names.append(i+str(self.permutation))
                    elif j==6:
                        self.redundance.append(int(i))
                    else:
                        self.recurrent.append(int(i))
                    
            self.number_actuators = int(info[8])
            self.min_freq = int(info[9])
            self.max_freq = int(info[10])
            self.codify = info[11]
            self.PMax = int(info[12])
            self.Pmin = int(info[13])
            self.step = int(info[14])
            self.time_network = int(info[15])
            self.dt = int(info[16])
            self.test = int(info[17]) == 1
            self.PI_test = int(info[18]) == 1
            
            self.control_signals=[self.hyperparam[0]*self.hyperparam[1],2*self.hyperparam[7]]
            self.neur = [[self.control_signals[0]*self.redundance[0],self.number_actuators],
                         [self.control_signals[1]*self.redundance[1],self.number_actuators]]      
        except:
            print("Error loading config file")
        
            
    def config_environment(self,rudder_ctrl,sails_ctrl):
        
        self.sail_env = env.sailboat_environment(rudder_ctrl = rudder_ctrl, sail_ctrl = sails_ctrl,
                                                 vmax = self.vmax, vmin = self.vmin, 
                                                 hyperparam = self.hyperparam, path = self.direction, 
                                                 learning = not self.test)
        
    def config_SNN_train(self):
        
        rudder_ctrl = SNN.spiking_neuron(controller='rudder', 
                                         name = self.files_names[0], 
                                         time_network = self.time_network, 
                                         redundance = self.redundance[0],
                                         max_out = self.AM[0], 
                                         min_out = self.Am [0],
                                         exit_state = self.hyperparam[3])
        
        sails_ctrl = SNN.spiking_neuron(controller='sail', 
                                        name = self.files_names[1], 
                                        time_network = self.time_network, 
                                        redundance = self.redundance[1],
                                        max_out = self.AM[1], 
                                        min_out = self.Am [1],
                                        exit_state = self.hyperparam[8])
        
        rudder_ctrl.SNN_setup(neurons=self.neur[0], dt=self.dt, time=self.time_network, 
                              recurrent=self.recurrent[0], model='LIF', wM = self.PMax, 
                              wm = self.Pmin, n = self.step, codify = self.codify, 
                              maximum = self.max_freq, minimun = self.min_freq)
            
        sails_ctrl.SNN_setup(neurons=self.neur[1], dt=self.dt, time=self.time_network, 
                             recurrent=self.recurrent[1], model='LIF', wM = self.PMax,
                             wm = self.Pmin, n = self.step, codify = self.codify, 
                             maximum = self.max_freq, minimun = self.min_freq)
        
        self.config_environment(rudder_ctrl = rudder_ctrl, sails_ctrl = sails_ctrl)
        
    def config_SNN_test(self):
        
        rudder_ctrl = SNN.spiking_neuron(name = self.files_names[0])     
        sails_ctrl = SNN.spiking_neuron(name = self.files_names[1])
        rudder_ctrl.load_SNN(path=self.direction,learning = not self.test)
        sails_ctrl.load_SNN(path=self.direction,learning = not self.test)
        self.config_environment(rudder_ctrl = rudder_ctrl, sails_ctrl = sails_ctrl)
        
    def train_SNN(self):
        
        self.change_simulation_type(2)
        self.config_SNN_train()    
        self.sail_env.set_database(db_name = 'Train_'+str(self.permutation), path = self.direction, 
                                   structure = ['scenario','state','reward_rudder',
                                                'reward_sail','epoch','wind','x','y',
                                                'speed'])
        band=False
        fail = False
        while self.sail_env.scenario <= 2 and not fail:
            if not band:
                band=self.p.open_port()
            else:
                data=self.p.read_data_sensor()
                if not isinstance(data,bool):
                    control_action = self.sail_env.environment_step(data = data, max_rate = self.max_freq,
                                                                    min_rate = self.min_freq)
                    self.p.write_control_action(control_action)
                    #self.sail_env.controllers[0].print_spikes(spike_ims=None, spike_axes=None)
                    fail = self.failure(name = 'Train_'+str(self.permutation), number = 4500)
                    if fail:
                        self.failure(name = 'Test_'+str(self.permutation), number = -1)
                else:
                    print("No data")
        self.p.write_control_action([0,0,0,1000])
        return fail
        
    def test_SNN(self):
        self.change_simulation_type(1)
        self.config_SNN_test()
        self.sail_env.set_database(db_name = 'Test_'+str(self.permutation), path = self.direction, 
                                   structure = ['state','wind','x','y','speed',
                                                'heeling','rudder','sail'])
        band=False
        fail = False
        while self.sail_env.state < len(self.sail_env.waypoints)-1 and not fail:
            if not band:
                band=self.p.open_port()
            elif band != 2:
                band = 2
                self.p.write_control_action([0,0,0,1000])
            else:
                data=self.p.read_data_sensor()
                if not isinstance(data,bool):
                    control_action = self.sail_env.environment_test(data = data, max_rate = self.max_freq,
                                                                    min_rate = self.min_freq)
                    self.p.write_control_action(control_action)
                    fail = self.failure(name = 'Test_'+str(self.permutation), number = 2500)
                else:
                    print("No data")
                    
    def test_PI(self):
        self.change_simulation_type(1)
        self.config_SNN_test()
        self.sail_env.set_database(db_name = 'TestPI', path = self.direction, 
                                   structure = ['state','wind','x','y','speed',
                                                'heeling'])
        band=False
        while self.sail_env.state < len(self.sail_env.waypoints)-1:
            if not band:
                band=self.p.open_port()
            elif band != 2:
                band = 2
                self.p.write_control_action([0,0,0,1000])
            else:
                self.sail_env.environment_PI_test(self.p)
        
        self.p.write_control_action([0,0,0,1000])
        
    def test_Viel2019(self):
        self.change_simulation_type(1)
        self.config_SNN_test()
        self.sail_env.set_database(db_name = 'TestViel2019', path = self.direction, 
                                   structure = ['state','wind','x','y','speed',
                                                'heeling'])
        band=False
        while self.sail_env.state < len(self.sail_env.waypoints)-1:
            if not band:
                band=self.p.open_port()
            elif band != 2:
                band = 2
                self.p.write_control_action([0,0,0,1000])
            else:
                data=self.p.read_data_sensor()
                if not isinstance(data,bool):
                    control_action = self.sail_env.environment_Viel2019_test(data = data)
                    self.p.write_control_action(control_action)
                else:
                    print("No data")
                    
    def test_SNN_real(self):
        i = 0
        self.change_simulation_type(1)
        self.config_SNN_test()
        while self.is_created(self.direction+'/data_result/Test_real_'+str(i)+'.csv'):
            i += 1
        self.sail_env.set_database(db_name = 'Test_real_'+str(i), path = self.direction, 
                                   structure = ['state','wind_angle','lat','lon','speed',
                                                'heading','rudder','sail','wind_speed'])
        band=False
        fail = False
        while self.sail_env.state < len(self.sail_env.waypoints)-1 and not fail:
            if not band:
                band=self.p.open_port()
            else:
                data=self.p.read_data_sensor()
                if not isinstance(data,bool):
                    control_action = self.sail_env.environment_test_real(data = data, max_rate = self.max_freq,
                                                                    min_rate = self.min_freq)
                    self.p.write_control_action(control_action)
                else:
                    print("No data")
        
    def change_database_number(self, number):
        self.sail_env.file_number = number
        
    def change_simulation_type(self, type_simulation):
        
        if type_simulation == 0:
            self.PI_test = True
            self.test = False
        elif type_simulation == 1:
            self.test = True
            self.PI_test = False
        else:
            self.test = False
            self.PI_test = False
            
    def failure(self,name,number):
        fail = False
        if self.sail_env.base.num_data>number:
            file = open(self.direction+'/data_result/'+name+'.csv', 'w')
            file.write('fail')
            file.close()
            fail=True
        return fail

    def is_created(self,name):
        filePath = name
        try:
            with open(filePath, 'r') as f:
                return True
        except:
            return False 
