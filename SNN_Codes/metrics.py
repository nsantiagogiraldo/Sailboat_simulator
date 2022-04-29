#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:37:28 2022

@author: nelson
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
#import pareto as prt

def files_list(num_experiments):
    test_files = []
    train_files = []
    for i in range(num_experiments):
        test_files.append('Test_'+str(i+1)+'.csv')
        train_files.append('Train_'+str(i+1)+'.csv')
    return test_files,train_files

def fail_list(test_files):
    fail = []
    not_fail = []
    for file in test_files:
        f = open(path+'/'+file,'r')
        info = f.read().split('\n')
        if info[0] == 'fail':
           fail.append(file)
        else:
            not_fail.append(file)
    return fail,not_fail

def graph_times(test_files,path):
    x = np.arange(0)
    y = np.arange(0)
    i = 1
    for file in test_files:
        f = open(path+'/'+file,'r')
        info = f.read().split('\n')
        time = int(info[-2].split(',')[0])
        x = np.append(x,i)
        y = np.append(y,time)
        i+=1
    return x,y

def config_plot(data,axisX,axisY,title,graph_type,legend=''):
    plt.figure(figsize=(10,7))
    plt.title(title)
    plt.xlabel(axisX)
    plt.ylabel(axisY)
    if graph_type == 'points':
        plt.plot(data[0],data[1],'o')
    elif graph_type == 'Vbar':
        plt.bar(data[0],data[1])
        for i in range(len(data[1])):
            plt.annotate(str(data[1][i]),(i-0.3,data[1][i]+1))
    elif graph_type == 'linear':
        if legend == '':
            plt.plot(data[0],data[1])
        else:
            plt.plot(data[0],data[1],label=legend)
    elif graph_type == 'linear_points':
        plt.annotate('', xy=(215, 90), xytext=(210, 90),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
        for i in range(len(data[1])-1):
            if i!=6 and i!=0:
                plt.annotate(str(i+1),(data[0][i]-0.3,data[1][i]+0.5))
            else:
                plt.annotate(str(i+1),(data[0][i]-1+(1/6)*i,data[1][i]+0.5))
        plt.annotate('Real Wind', xy=(210, 91))
        plt.plot(data[0],data[1],data[0],data[1],'o')
    elif graph_type == 'linear_points2':
        plt.annotate('', xy=(220, 85), xytext=(215, 85),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
        for i in range(len(data[1])-1):
            if data[0][i] == 240 and data[1][i] == 100:
                plt.annotate('Origin', xy=(data[0][i]-1.4,data[1][i]+0.5))
            else:
                plt.annotate(str((i+1)//2),(data[0][i]-0.3,data[1][i]+0.5))
        plt.annotate('Real Wind', xy=(215, 86))
        plt.plot(data[0],data[1],data[0],data[1],'o')
        
def config_subplots(num,data,axisX,axisY,title,graph_type,x=0):
    dim = int(np.ceil(np.sqrt(num)))
    fig, axs = plt.subplots(dim,dim)
    fig.set_size_inches(10,7)
    if graph_type == 'points' or graph_type == 'line':
        x = np.array(range(np.size(data[0])))+1
    
    for i in range(dim):
        for j in range(dim):
            axs[i, j].set_title(title[dim*i+j])
            if graph_type == 'points':
                axs[i, j].plot(x, data[dim*i+j], 'o')
            elif graph_type == 'line':
                axs[i, j].plot(x, data[dim*i+j])
            elif graph_type == 'Vbar':
                axs[i, j].bar(x[dim*i+j],data[dim*i+j])
                for k in range(len(data[dim*i+j])):
                    axs[i, j].annotate(str(data[dim*i+j][k]),(k-0.3,data[dim*i+j][k]+1))         
    for ax in axs.flat:
        ax.set(xlabel=axisX, ylabel=axisY)
    fig.tight_layout()
    return fig

def config_subplots2(num,data,axisX,axisY,title,graph_type,x=0):
    dim = int(np.ceil(np.sqrt(num)))
    fig, axs = plt.subplots(1,dim)
    fig.set_size_inches(10,7)
    if graph_type == 'points' or graph_type == 'line':
        x = np.array(range(np.size(data[0])))+1
    
    for j in range(dim):
        axs[j].set_title(title[j])
        if graph_type == 'points':
            axs[j].plot(x, data[j], 'o')
        elif graph_type == 'line':
            axs[j].plot(x, data[j])
        elif graph_type == 'Vbar':
            axs[j].bar(x[j],data[j])
            for k in range(len(data[+j])):
                axs[j].annotate(str(data[j][k]),(k-0.3,data[j][k]+1))         
    for ax in axs.flat:
        ax.set(xlabel=axisX, ylabel=axisY)
    fig.tight_layout()
    return fig
        
def config_bar_data(data,start,finish,intervals):
    limits = [start]
    x = []
    step = (finish - start)/intervals
    while limits[-1]<finish-0.1:
        limits.append(limits[-1]+step)
    y, xn = np.histogram(data, bins=limits)
    for i in range(len(xn)-1):
        x.append(str((round(xn[i],1))))
    return x,y

def environment_points(test):
    initial_point = [240, 100] 
    waypoints = [[],[]]
    points = 6
    dist = 14
    if test:
        band = True
        center = [0,0]
        phi = 360/points
        x = dist
        center[0] =  initial_point[0]-x
        center[1] =  initial_point[1]
        r = np.sqrt((initial_point[0]-center[0])**2+(initial_point[1]-center[1])**2)         
        
        n = 2*points
        theta = 0
        
        for i in range(n-1):
            if band and i>= points:
                band = False
                theta = 180 
                center[0] = initial_point[0]+x
                phi *= -1                   
                
            theta += phi
            xf = int(r*np.cos(np.radians(theta))+center[0])
            yf = int(r*np.sin(np.radians(theta))+center[1])
            waypoints[0].append(xf)
            waypoints[1].append(yf)
    else:
        center = initial_point
        n = 10
        theta = np.pi/2
        r = 20
        
        for i in range(n):           
            x = int(r*np.cos(theta)+center[0])
            y = int(r*np.sin(theta)+center[1])
            waypoints[0].append(x)
            waypoints[1].append(y)
            waypoints[0].append(center[0])
            waypoints[1].append(center[1])
            theta -= np.pi/(n-1)
 
        center = initial_point # Turn the wind. Tacking
        n = 3
        theta=np.pi/2
        r = 25
        for i in range(n):        
            
            theta += np.pi/(n+1)
            x = int(r*np.cos(theta)+center[0])
            y = int(r*np.sin(theta)+center[1])
            waypoints[0].append(x)
            waypoints[1].append(y)
            waypoints[0].append(center[0])
            waypoints[1].append(center[1])
        
    waypoints[0].insert(0, initial_point[0])
    waypoints[1].insert(0, initial_point[1])
    waypoints[0].insert(len(waypoints[0]), initial_point[0])
    waypoints[1].insert(len(waypoints[1]), initial_point[1])
    

    return waypoints

def rect_equiation(x,y):
    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[1] - m*x[1]
    
    return [m,b]

def error_data(file,path):
    x = np.arange(0)
    y = np.arange(0)
    error = np.arange(0)
    final_error = np.arange(0)
    scenario = [1,1]
    f = open(path+'/'+file,'r')
    info = f.read().split('\n')
    env_points = environment_points(test=True)
    ideal = rect_equiation(env_points[0][0:2], env_points[1][0:2])
    for i in range(len(info)-2):
        split_info = info[i+1].split(',',2)
        scenario[1] = int(split_info[1])
        if (scenario[1]+1) % 3 != 0:
            if scenario[1] != scenario[0]:
                real = ideal[0]*x+ideal[1]
                error = np.append(error,error_metric(real = real, pred = y, error_type = 'MAE'))
                x = np.arange(0)
                y = np.arange(0)
                ideal = rect_equiation(env_points[0][scenario[0]:scenario[1]+1], env_points[1][scenario[0]:scenario[1]+1])
            data = split_info[2].split(',')
            x = np.append(x,float(data[1]))
            y = np.append(y,float(data[2]))
            
        scenario[0] = scenario[1]
    real = ideal[0]*x+ideal[1]
    error = np.append(error,error_metric(real = real, pred = y, error_type = 'MAE'))
    for i in range(4):
        final_error = np.append(final_error,np.mean([error[i],error[7-i]]))
    
    return final_error

def error_graphs(test_files,path):
    complete = error_data(test_files[0],path)
    for i in range(len(test_files)-1):
        complete = np.vstack((complete,error_data(test_files[i+1],path)))
    return complete
    
def error_metric(real, pred, error_type):
    ERROR = pred - real
    if error_type == 'MAE':
        ERROR = np.sum(np.abs(ERROR))/np.size(ERROR)
    elif error_type == 'MAPE':
        ERROR = np.sum(np.abs(ERROR)/real)/np.size(ERROR)
    elif error_type == 'MSE':
        ERROR = np.sum(ERROR*ERROR)/np.size(ERROR)
    elif error_type == 'MSPE':
        ERROR = np.sum(ERROR*ERROR/(real*real))/np.size(ERROR)
    return ERROR

def read_config_data(path,name_test):
    num = name_test[5:-4]
    file = 'config'+str(num)+'.txt'
    f = open(path+'/'+file,'r')
    info = f.read().split('\n')
    info = info[4].split(',')
    f.close()
    
    return [float(info[0]),float(info[7]),float(info[3]),float(info[8])] #3 y 8 tambien son importantes

def train_simple_data(path,name):
    f = open(path+'/'+name,'r')
    info = f.read().split('\n')
    time = int(info[-3].split(',')[0])
    epochs = int(info[-3].split(',')[5]) - 2
    f.close()
    return [time,epochs]

def train_reward(path,name):
    rudder = []
    sail = []
    f = open(path+'/'+name,'r')
    info = f.read().split('\n')
    for i in range(len(info)-3):  
        rudder.append(float(info[i+1].split(',')[3]))
        sail.append(float(info[i+1].split(',')[4]))
    return rudder,sail

def test_curve(path,name):
    x = []
    y = []
    f = open(path+'/'+name,'r')
    info = f.read().split('\n')
    for i in range(len(info)-4):  
        x.append(float(info[i+2].split(',')[3]))
        y.append(float(info[i+2].split(',')[4]))
    return x,y

def test_param(path,name,param):
    x = []
    y = []
    f = open(path+'/'+name,'r')
    info = f.read().split('\n')
    for i in range(len(info)-4):  
        y.append(float(info[i+2].split(',')[param]))
        x.append(i+1)
    return x,y

def train_curve(path,name):
    x = []
    y = []
    f = open(path+'/'+name,'r')
    info = f.read().split('\n')
    for i in range(len(info)-4):  
        x.append(float(info[i+2].split(',')[7]))
        y.append(float(info[i+2].split(',')[8]))
    return x,y
path = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/results/data'
path2 = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/config'
images = ['graph_times.png','histogram_times.png','Test_obj_points.png','Error_graphs.png',
          'Histogram_errors.png','Reward_change.png', 'Test_curve.png', 'Train_curve.png',
          'Test_rudder_action.png', 'Test_sail_action.png', 'Train_obj_points.png']
image = 10
num_experiments = 1024
choosen = 923
train = 'Train_'+str(choosen)+'.csv'
test_choosen = 'Test_'+str(choosen)+'.csv'

test_files,train_files = files_list(num_experiments)
fail_test, not_fail_test = fail_list(test_files)
fail_train, not_fail_train = fail_list(train_files)
print('Total fail test scenarios: ',  num_experiments-len(not_fail_test)-len(fail_train), ' %:', (num_experiments-len(not_fail_test)-len(fail_train))*100/num_experiments)
print('Total not fail test scenarios: ', len(not_fail_test), ' %:', len(not_fail_test)*100/num_experiments)
print('Total fail train scenarios: ', len(fail_train), ' %:', len(fail_train)*100/num_experiments)

try:
    if image == 0:
        axisX = 'Finished test scanarios'
        axisY = 'Test time'
        title = 'Test time for each finished test experiment' 
        x,y = graph_times(not_fail_test,path)
        config_plot([x,y], axisX, axisY, title, 'points')
    elif image == 1:
        axisX = 'Test times'
        axisY = 'Number of experiments'
        title = 'Number of test experiments for each time interval'
        x,y = graph_times(not_fail_test,path)
        x,y = config_bar_data(y,300,2300,20)
        config_plot([x,y], axisX, axisY, title, 'Vbar')
    elif image == 2:
        axisX = 'X'
        axisY = 'Y'
        title = 'Test objective points'
        obj_points = environment_points(True)
        x = np.array(obj_points[0])
        y = np.array(obj_points[1])
        config_plot([x,y], axisX, axisY, title, 'linear_points')
        pts = environment_points(True)
        arrow=patch.ArrowStyle.Fancy(head_length=.8, head_width=.8, tail_width=.001)
        for i in range(len(pts[0])-1):
            plt.annotate('', xy=((pts[0][i+1]+pts[0][i])/2, (pts[1][i+1]+pts[1][i])/2), xytext=(pts[0][i], pts[1][i]),
                          arrowprops=dict(arrowstyle=arrow,color=[37/255,150/255,190/255])
                        )
    elif image == 3:
        axisX = 'Number of designing point'
        axisY = 'MAE'
        title = ['MAE for navigation type 1','MAE for navigation type 2',
                 'MAE for navigation type 3','MAE for navigation type 4']
        error = error_graphs(not_fail_test, path)
        error = np.transpose(error)
        fig = config_subplots(4,error,axisX,axisY,title,'points')
        fig.savefig(images[image])
    elif image == 4:
        inf = [0,0,0,0]
        sup = [4,5,6,5]
        step = [14,14,14,14]
        axisX = 'MAPE'
        axisY = 'Number of experiments'
        title = ['Number of test experiments for each MAE (type 1)',
                 'Number of test experiments for each MAE (type 2)',
                 'Number of test experiments for each MAE (type 3)',
                 'Number of test experiments for each MAE (type 4)']
        error = error_graphs(not_fail_test, path)
        error = np.transpose(error)
        x,y = config_bar_data(error[0],inf[0],sup[0],step[0])
        xa = np.array(x)
        ya = np.array(y)
        for i in range(3):
            x,y = config_bar_data(error[i+1],inf[i+1],sup[i+1],step[i+1])
            xa = np.append(xa,np.array(x))
            ya = np.append(ya,np.array(y))
        xa.resize(4,14)
        ya.resize(4,14)
        fig = config_subplots(4,ya,axisX,axisY,title,'Vbar',xa)
        fig.savefig(images[image])
        
    # elif image == 5:
    #     paret = prt.Pareto_optimization(maximization=False)
    #     x,y = graph_times(not_fail_test,path)
    #     error = error_graphs(not_fail_test, path)
    #     error = np.transpose(error)
    #     metric = np.concatenate([[y],error])        
    #     points = paret.pareto_optimum(metric)
    #     new_test = []
        
    #     for i in points:
    #         new_test.append(not_fail_test[i])  
            
    #     pts = np.arange(0)
    #     for fi in new_test:
    #         data = [read_config_data(path2,fi)]
    #         if len(pts) == 0:
    #             pts = np.array(data)
    #         else:
    #             pts = np.concatenate([pts,data])
    #     pts = np.transpose(pts)
    #     points = paret.pareto_optimum(pts)
    #     pts = np.transpose(pts)
    #     error = np.transpose(error)
    #     for i in points:
    #         j = not_fail_test.index(new_test[i])
    #         print(not_fail_test[j], 'T', y[j], 'Error', error[j],
    #               'rend',pts[i])
    elif image == 6:
        info = train_simple_data(path, train)
        print('Training time: ', info[0],'\n','Epochs:',info[1])
        axisX = 'Epochs'
        axisY = 'Reward'
        title = ['Rudder training reward',
                 'Sail training reward']
        rudder,sail = train_reward(path,train)
        config_subplots2(2,[rudder,sail],axisX,axisY,title,'line')
        
    elif image == 7:
        axisX = 'X'
        axisY = 'Y'
        title = 'Sailboat test curve' 
        x,y =  test_curve(path,test_choosen)
        obj_points = environment_points(True)
        x1 = np.array(obj_points[0])
        y1 = np.array(obj_points[1])
        config_plot([x,y], axisX, axisY, title, 'linear','Real test curve')
        plt.annotate('', xy=(215, 85), xytext=(210, 85),
             arrowprops=dict(facecolor='black', shrink=0.08),
             )
        plt.annotate('Real Wind', xy=(210, 87))
        obj_points = environment_points(True)
        pts = obj_points
        arrow=patch.ArrowStyle.Fancy(head_length=.8, head_width=.8, tail_width=.001)
        for i in range(len(pts[0])-1):
            plt.annotate('', xy=((pts[0][i+1]+pts[0][i])/2, (pts[1][i+1]+pts[1][i])/2), xytext=(pts[0][i], pts[1][i]),
                          arrowprops=dict(arrowstyle=arrow,color='g')
                        )
        x = np.array(obj_points[0])
        y = np.array(obj_points[1])
        plt.plot(x,y,'g',label = 'Ideal test curve')
        plt.plot(x,y,'o',color='g')
        plt.legend()
        
        
    elif image == 8:
        axisX = 'X'
        axisY = 'Y'
        title = 'Sailboat train curve' 
        x,y =  train_curve(path,train)
        config_plot([x,y], axisX, axisY, title, 'linear')
        plt.annotate('', xy=(222, 85), xytext=(218, 85),
             arrowprops=dict(facecolor='black', shrink=0.08),
             )
        plt.annotate('Real Wind', xy=(218, 87))
    
    elif image == 9:
        axisX = 'Time'
        axisY = 'Control Action'
        title = 'Rudder control action' 
        x,y =  test_param(path,test_choosen,7)
        x = x[169:207]
        y = y[169:207]
        config_plot([x,y], axisX, axisY, title, 'linear')
    
    elif image == 10:
        axisX = 'Time'
        axisY = 'Control Action'
        title = 'Sail control action' 
        x,y =  test_param(path,test_choosen,8)
        x = x[169:207]
        y = y[169:207]
        config_plot([x,y], axisX, axisY, title, 'linear')
        
    elif image == 11:
        axisX = 'X'
        axisY = 'Y'
        title = 'Train objective points'
        obj_points = environment_points(False)
        x = np.array(obj_points[0])
        y = np.array(obj_points[1])
        config_plot([x,y], axisX, axisY, title, 'linear_points2')
    elif image == 12:
        print(error_data('TestViel2019.csv',path))
        print(error_data('Test_923.csv',path))
    if image <= 2:
        plt.savefig(images[image])
    elif image >= 6 and image !=12:
        plt.savefig(images[image-1])
    
    
except:
    
    print('Error')