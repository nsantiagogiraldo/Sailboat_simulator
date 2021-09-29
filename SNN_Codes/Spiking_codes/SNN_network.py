import torch
import bindsnet.analysis.plotting as plt
import bindsnet.network.nodes as nodes
from bindsnet.network.network import Network #Permite crear un nuevo objeto de la clase red
from bindsnet.learning import MSTDP
import bindsnet.encoding.encodings as codify
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network.network import load

def max_min_spikes(neuronas, dt, time, recurrent, modelo, wM, wm, n, codify, maximum, minimun):
    data=[]
    data1=[]
    [capas,monitores,net,m_pesos] = arquitectura_red(neuronas,dt,time,recurrent,modelo,wM,wm,n,ones=True)
    for i in range(neuronas[0]):
        data.append(maximum)
        data1.append(minimun)
    #data[0]=0
    #data[1]=0
    #data[2]=0
    #data[3]=0
    #data1[0]=1
    #data1[1]=1
    #data1[2]=1
    #data1[3]=1
    dato = codificacion_red(codify,time,dt,data)
    dato2 = codificacion_red(codify,time,dt,data1)
    net.run(inputs={capas[0] : dato}, time=time,reward=0)
    imprimir_spikes(monitores,time,None,None)
    h=monitores[len(capas)-1].get("s")
    h=h.sum(dim=0).tolist()[0]
    net.run(inputs={capas[0] : dato2}, time=time,reward=0)
    imprimir_spikes(monitores,time,None,None)
    c=monitores[len(capas)-1].get("s")
    c=c.sum(dim=0).tolist()[0]
    
    return h[0],c[0]
    
def imprimir_spikes(spikes,time,spike_ims, spike_axes): 
    _spikes={}
    for i, l in enumerate(spikes):
        _spikes[i] = spikes[l].get("s").view(time, -1)
    spike_ims, spike_axes = plt.plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
    plt.plt.show()
    return spike_ims, spike_axes

def imprimir_pesos(weigths, wmax, wmin, im):
    _weigths={}
    for i, l in enumerate(weigths):
        _weigths[i] = weigths[l].get("w")
    weigths_ims = plt.plot_weights(_weigths[1][1],im=im,wmin=wmin,wmax=wmax)
    plt.plt.show()
    return weigths_ims
    
def guardar_red(name, network, direction, max_sp, min_sp, arq):
    j=True
    try:
        network.save(direction+'/'+name)
        arq_str = str(arq)
        f = open('network_connection.txt', 'w')
        f.write(str(max_sp)+'\n'+str(min_sp)+'\n'+arq_str[1:len(arq_str)-1])
        f.close()
    except:
        j=False
        
    return j

def cargar_red(name, direction, learning):
    spike_monitors={}
    weight_monitors={}
    wc=0
    sc=0
    arq=[]
    try:
        network = Network() 
        network = load(direction+'/'+name,learning=learning)
        monitors=network.monitors;
        for i in monitors:
            if i[0]=='S':
                spike_monitors[sc]=monitors[i]
                sc+=1
            else:
                weight_monitors[wc]=monitors[i]
                wc+=1
        capas = list(network.layers.keys())
                
        f = open('network_connection.txt', 'r')
        info = f.read().split('\n')
        max_spikes = int(info[0])
        min_spikes = int(info[1])
        for i in info[2].split(','):
            arq.append(int(i))
        
        # for i in range(k):
        #     M_weigths[i] = Monitor(k[i], state_vars='w')
        # k=network.

    except:
        print("Error loading")
        
    return capas,spike_monitors,network,weight_monitors, max_spikes, min_spikes, arq

def codificacion_red(code, time_network, dt, dato):
    dato=torch.Tensor(dato)
    if code=='bernoulli':
        t=codify.bernoulli(dato,time=time_network, dt=dt)
    elif code=='poisson':
        t=codify.poisson(dato,time=time_network, dt=dt)
    elif code=='rank_order':
        t=codify.rank_order(dato,time=time_network, dt=dt)
    elif code=='repeat':
        t=codify.repeat(dato,time=time_network, dt=dt)
    else:
        t=codify.single(dato,time=time_network, dt=dt)
    return t

def arquitectura_red(neuronas, dt, time, recurrent, modelo, wM, wm, n, ones=False):
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
            neur[i] =tipo_neurona(modelo, neuronas[i])                
            network.add_layer(neur[i], name=label[i])
            # if i==1:
            #     connect[i-1] = Connection(source=neur[i-1],target=neur[i],wmin=0,wmax=1e-1)
            # else:
            if ones:
                connect[i-1] = Connection(
                    source=neur[i-1],
                    target=neur[i],
                    wmin=0,
                    wmax=1,
                    update_rule=MSTDP,
                    nu=n,
                    w=(wM-wm)*torch.ones(neur[i-1].n, neur[i].n)+wm,
                    #w=0.05 + 0.1 * torch.randn(neur[i-1].n, neur[i].n)
                    #norm=0.5 * neur[i-1].n,
                )
            elif i!=1:
                connect[i-1] = Connection(
                    source=neur[i-1],
                    target=neur[i],
                    wmin=wm,
                    wmax=wM,
                    update_rule=MSTDP,
                    nu=n,
                    w=(wM-wm)*torch.randn(neur[i-1].n, neur[i].n)+wm,
                    #w=0.05 + 0.1 * torch.randn(neur[i-1].n, neur[i].n)
                    #norm=0.5 * neur[i-1].n,
                )
            else:
                connect[i-1] = Connection(
                    source=neur[i-1],
                    target=neur[i],
                    wmin=wm,
                    wmax=wM,
                    #update_rule=MSTDP,
                    #nu=n,
                    #w=(wM-wm)*torch.randn(neur[i-1].n, neur[i].n)+wm,
                    #w=0.05 + 0.1 * torch.randn(neur[i-1].n, neur[i].n)
                    #norm=0.5 * neur[i-1].n,
                )
            network.add_connection(connect[i-1], source=label[i-1], target=label[i])
            M_weigths[i-1] = Monitor(connect[i-1], state_vars='w', time=time)
            network.add_monitor(M_weigths[i-1], name="Pesos capa %s" % label[i])
            if i == len(neuronas)-1 and recurrent !=0 and ones:
                connect[i] = Connection(
                    source=neur[i],
                    target=neur[recurrent],
                    #w=0.025 * (torch.eye(neur[recurrent].n) - 1),
                    wmin=wm,
                    wmax=wM,
                    update_rule=MSTDP,
                    nu=n,
                    w=(wM-wm)*torch.randn(neur[i].n, neur[recurrent].n)+wm
                    #norm=0.5 * neur[i].n,
                )
                network.add_connection(connect[i], source=label[i], target=label[recurrent])
            elif i == len(neuronas)-1 and recurrent !=0:
                connect[i] = Connection(
                    source=neur[i],
                    target=neur[recurrent],
                    #w=0.025 * (torch.eye(neur[recurrent].n) - 1),
                    wmin=wm,
                    wmax=wM,
                    update_rule=MSTDP,
                    nu=n,
                    w=(wM-wm)*torch.randn(neur[i].n, neur[recurrent].n)+wm
                    #norm=0.5 * neur[i].n,
                )
                network.add_connection(connect[i], source=label[i], target=label[recurrent])
        M_spikes[i] = Monitor(neur[i], state_vars='s', time=time)
        network.add_monitor(M_spikes[i], name="Spikes capa %s" % label[i])
        
    return label,M_spikes,network,M_weigths
                
def tipo_neurona(modelo,neuronas):
        if modelo == 'LIF':
            neur = nodes.LIFNodes(neuronas,traces=True)
        elif modelo =='AdaptiveLIF':
            neur = nodes.AdaptiveLIFNodes(neuronas,traces=True)
        elif modelo == 'BoostedLIF':
            neur = nodes.BoostedLIFNodes(neuronas,traces=True)
        elif modelo == 'CSRM':
            neur = nodes.CSRMNodes(neuronas,traces=True)
        elif modelo == 'CurrentLIF':
            neur = nodes.CurrentLIFNodes(neuronas,traces=True)
        elif modelo == 'DiehlAndCook':
            neur = nodes.DiehlAndCookNodes(neuronas,traces=True)
        elif modelo == 'IF':
            neur = nodes.IFNodes(neuronas,traces=True)
        else:
            neur = nodes.IzhikevichNodes(neuronas,traces=True)
        
        return neur
    

def normalize(data,vmax,vmin, A=1, B=0):
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

def decode_spikes(monitor, spikes, Act_M, Act_m, err_dist):
    h=monitor.get("s")
    h=h.sum(dim=0).tolist()[0]
    
    actions = []
    for i in range(len(h)):
        a = normalize(data=[h[i]], vmax=[spikes[0]-5], vmin=[spikes[1]+5], A=Act_M[i], B=Act_m[i])
        actions = actions+a
    actions.append(0)
    if actions[1]>0:
        actions[1]=90
    else:
        actions[1]=-90
    if err_dist<2:
        final = 1
    else:
        final = 0
    
    return actions+[final]