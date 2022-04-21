import torch
import bindsnet.analysis.plotting as plt
import bindsnet.network.nodes as nodes
import copy as cp
import bindsnet.encoding.encodings as codify
from bindsnet.network.network import Network #Permite crear un nuevo objeto de la clase red
from bindsnet.learning import MSTDP
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.network import load

class spiking_neuron:
    
    spiking_controller={}
    spiking_monitors = {}
    weigths_monitors = {}
    layers = {}
    network_architecture = {}
    max_spikes = 0
    redundance = 0
    min_spikes = 0
    is_rudder_controller = True
    time_network = 0
    max_weigth = 0
    min_weigth = 0
    network_name = ''
    coding = ''
    dt = 1
    max_out = 0
    min_out =0
    out_states = 0
    
    def __init__(self, name, max_out = 0, min_out = 0, exit_state = 0, 
                 redundance = 0, time_network = 0, controller=0, train = True):
        if train:
            if controller == "rudder":
                self.is_rudder_controller = True
            else:
                self.is_rudder_controller = False
            self.time_network = time_network
            self.network_name = name
            self.redundance = redundance
            self.max_out = max_out
            self.min_out = min_out
            self.out_states = exit_state
            
    def SNN_setup(self, neurons, dt, time, recurrent, model, wM, wm, n, codify, maximum, minimun):
        data=[]
        data1=[]
        neur = cp.copy(neurons)
        neur[0] = self.redundance
        self.max_weigth = wM
        self.min_weigth = wm
        self.time_network = time
        self.dt = dt
        self.coding = codify
        self.network_architecture = neurons
        
        [layers,monitors,net,m_pesos] = self.SNN_architecture(neur,dt,time,recurrent,model,wM,wm,n,ones=True)
        for i in range(neur[0]):
            data.append(maximum)
            data1.append(minimun)
        dato = self.SNN_encoding(data,1)
        dato2 = self.SNN_encoding(data1,1)
        net.run(inputs={layers[0] : dato}, time=time,reward=0)
        #self.print_spikes(None,None)
        h=monitors[len(layers)-1].get("s")
        h=h.sum(dim=0).tolist()[0]
        net.run(inputs={layers[0] : dato2}, time=time,reward=0)
        #self.print_spikes(None,None)
        c=monitors[len(layers)-1].get("s")
        c=c.sum(dim=0).tolist()[0]
        
        [layer,monitors,net,m_pesos] = self.SNN_architecture(neurons,dt,time,recurrent,model,wM,wm,n,ones=False)
        
        self.max_spikes = h[0]
        self.min_spikes = c[0]
        self.layers = layer
        self.spiking_monitors = monitors
        self.spiking_controller = net
        self.weigths_monitors = m_pesos
 
        
    def print_spikes(self,spike_ims, spike_axes): 
        _spikes={}
        for i, l in enumerate(self.spiking_monitors):
            _spikes[i] = self.spiking_monitors[l].get("s").view(self.time_network, -1)
        spike_ims, spike_axes = plt.plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
        plt.plt.show()
        return spike_ims, spike_axes
    
    def print_weigths(self, im):
        _weigths={}
        for i, l in enumerate(self.weigths_monitors):
            _weigths[i] = self.weigths_monitors[l].get("w")
        weigths_ims = plt.plot_weights(_weigths[len(_weigths)-1][1],im=im,wmin=self.min_weigth,wmax=self.max_weigth)
        if self.is_rudder_controller:
            plt.plt.savefig('Rudder_weigths.png')
        else:
            plt.plt.savefig('Sail_weigths.png')
        plt.plt.show()
        return weigths_ims
        
    def save_SNN(self, path):
        j=True
        name_ctrl = ['sail','rudder']
        try:
            self.spiking_controller.save(path+'/control/'+self.network_name+".pt")
            arq_str = str(self.network_architecture)
            f = open(path+'/control/'+self.network_name + '_connection.txt', 'w')
            f.write(str(self.max_spikes)+'\n'+str(self.min_spikes)+'\n'+arq_str[1:len(arq_str)-1]+'\n'
                    +str(self.time_network)+'\n'+str(self.redundance)+'\n'+str(self.dt)+'\n'+self.coding+
                    '\n'+str(self.max_weigth)+'\n'+str(self.min_weigth)+'\n'+str(self.max_out)+'\n'+
                    str(self.min_out)+'\n'+str(self.out_states)+'\n'+name_ctrl[self.is_rudder_controller]+
                    '\n'+str(self.time_network)+'\n'+str(self.redundance))
            f.close()
        except:
            j=False
            
        return j
    
    def load_SNN(self, path, learning):
        spike_monitors={}
        weight_monitors={}
        wc=0
        sc=0
        arq=[]
        try:
            network = Network() 
            network = load(path+'/control/'+self.network_name+'.pt',learning=learning)
            monitors=network.monitors;
            for i in monitors:
                if i[0]=='S':
                    spike_monitors[sc]=monitors[i]
                    sc+=1
                else:
                    weight_monitors[wc]=monitors[i]
                    wc+=1
            layer = list(network.layers.keys())
                    
            f = open(path+'/control/'+self.network_name + '_connection.txt', 'r')
            info = f.read().split('\n')
            max_spikes = int(info[0])
            min_spikes = int(info[1])
            for i in info[2].split(','):
                arq.append(int(i))
            time = int(info[3])
            red = int(info[4])
            dt = int(info[5])
            coding = info[6]
            wMax = int(info[7])
            wMin = int(info[8])
            max_o = int(info[9])
            min_o = int(info[10])
            exit_states = int(info[11])
            name_nt = info[12]
            time = int(info[13])
            red = int(info[14])
            if name_nt == 'rudder':
                self.is_rudder_controller = True
            else:
                self.is_rudder_controller = False
            
            self.layers = layer
            self.spiking_monitors = spike_monitors
            self.spiking_controller = network
            self.weigths_monitors = weight_monitors
            self.max_spikes = max_spikes
            self.min_spikes = min_spikes
            self.network_architecture = arq
            self.time_network = time
            self.dt = dt
            self.redundance = red
            self.max_weigth =wMax
            self.coding = coding
            self.min_weigth = wMin
            self.max_out = max_o
            self.min_out = min_o
            self.out_states = exit_states
            self.time_network = time
            self.redundance = red

        except:
            print("Error loading")

    
    def SNN_encoding(self, dato, redundance):
        time = self.time_network
        code = self.coding
        data_redundant=[]
        for i in dato:
            data_redundant = data_redundant+redundance*[i]
        data_redundant=torch.Tensor(data_redundant)
        if code=='bernoulli':
            t=codify.bernoulli(data_redundant,time=time, dt=self.dt)
        elif code=='poisson':
            t=codify.poisson(data_redundant,time=time, dt=self.dt)
        elif code=='rank_order':
            t=codify.rank_order(data_redundant,time=time, dt=self.dt)
        elif code=='repeat':
            t=codify.repeat(data_redundant,time=time, dt=self.dt)
        else:
            t=codify.single(data_redundant,time=time, dt=self.dt)
        return t
    
    def SNN_architecture(self, neuronas, dt, time, recurrent, model, wM, wm, n, ones=False):
        network = Network(dt=dt, learning=True) 
        neur = {}
        connect={}
        label={}
        M_spikes={}
        M_weigths={}
        for i in range(len(neuronas)):
            if i==0:
                label[0]="Entrada"
            elif i==len(neuronas)-1:
                label[len(neuronas)-1] = "Salida"
            else:
                label[i] = "Oculta_"+str(i)
        
        for i in range(len(neuronas)):
            if i==0:
                neur[i] = nodes.Input(neuronas[i], traces=True)
                network.add_layer(neur[i], name=label[i])
            else:
                neur[i] = self.neuron_model(model, neuronas[i])                
                network.add_layer(neur[i], name=label[i])
                if ones:
                    connect[i-1] = Connection(
                        source=neur[i-1],
                        target=neur[i],
                        wmin=-wm,
                        wmax=wM,
                        update_rule=MSTDP,
                        nu=n,
                        w=(wM-wm)*torch.ones(neur[i-1].n, neur[i].n)+wm,
                    )
                else:
                    connect[i-1] = Connection(
                        source=neur[i-1],
                        target=neur[i],
                        wmin=wm,
                        wmax=wM,
                        update_rule=MSTDP,
                        nu=n,
                    )
                network.add_connection(connect[i-1], source=label[i-1], target=label[i])
                M_weigths[i-1] = Monitor(connect[i-1], state_vars='w', time=time)
                network.add_monitor(M_weigths[i-1], name="Pesos capa %s" % label[i])
                if i == len(neuronas)-1 and recurrent !=0 and ones:
                    connect[i] = Connection(
                        source=neur[i],
                        target=neur[recurrent],
                        wmin=wm,
                        wmax=wM,
                        update_rule=MSTDP,
                        nu=n,
                    )
                    network.add_connection(connect[i], source=label[i], target=label[recurrent])
                elif i == len(neuronas)-1 and recurrent !=0:
                    connect[i] = Connection(
                        source=neur[i],
                        target=neur[recurrent],
                        wmin=wm,
                        wmax=wM,
                        update_rule=MSTDP,
                        nu=n,
                    )
                    network.add_connection(connect[i], source=label[i], target=label[recurrent])
            M_spikes[i] = Monitor(neur[i], state_vars='s', time=time)
            network.add_monitor(M_spikes[i], name="Spikes capa %s" % label[i])
            
        return label,M_spikes,network,M_weigths
                    
    def neuron_model(self,model,neurons):
        
        if model == 'LIF':
            neur = nodes.LIFNodes(neurons,traces=True)
        elif model =='AdaptiveLIF':
            neur = nodes.AdaptiveLIFNodes(neurons,traces=True)
        elif model == 'BoostedLIF':
            neur = nodes.BoostedLIFNodes(neurons,traces=True)
        elif model == 'CSRM':
            neur = nodes.CSRMNodes(neurons,traces=True)
        elif model == 'CurrentLIF':
            neur = nodes.CurrentLIFNodes(neurons,traces=True)
        elif model == 'DiehlAndCook':
            neur = nodes.DiehlAndCookNodes(neurons,traces=True)
        elif model == 'IF':
            neur = nodes.IFNodes(neurons,traces=True)
        else:
            neur = nodes.IzhikevichNodes(neurons,traces=True)
        
        return neur       
    
    def normalize(self, data, vmax, vmin, A=1, B=0):
        
        fn=[]
        for i in range(len(data)):
            if data[i]<=vmin[i]:
                fn.append(B);
            elif data[i]>=vmax[i]:
                fn.append(A);
            else:
                k=(A*(data[i]-vmin[i])-B*(data[i]-vmax[i]))/(vmax[i]-vmin[i])
                fn.append(k);
        return fn
    
    def decode_spikes(self):
        
        actions = []
        h=self.spiking_monitors[len(self.network_architecture)-1].get("s")
        h=h.sum(dim=0).tolist()[0]
        spikes = [self.max_spikes, self.min_spikes]
             
        for i in range(len(h)):
            a = self.normalize(data=[h[i]], vmax=[spikes[0]-2], vmin=[spikes[1]+2], A=self.max_out, B=self.min_out)
            k = (self.max_out-self.min_out)/self.out_states
            state = int((a[0]+self.max_out)/k)
            if state>=self.out_states:
                state = self.out_states-1
            a[0] = int(((2*state+1)*k+2*self.min_out)//2)
            actions = actions+a
            
        return actions[0]
    
    def train_episode(self, n_data,reward):
        
        cod_n_dato = self.SNN_encoding(n_data, self.redundance)
        self.spiking_controller.run(inputs={self.layers[0] : cod_n_dato}, time=self.time_network, reward=reward)
        #self.print_spikes(spike_ims=None, spike_axes=None)
        control_action=self.decode_spikes()
        
        return control_action
    
    def reset_state_variables(self):
        self.spiking_controller.reset_state_variables()
