#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:37:28 2022

@author: nelson
"""

import numpy as np
import matplotlib.pyplot as plt


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

def config_plot(data,axisX,axisY,title,graph_type):
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
        plt.plot(data[0],data[1])
    elif graph_type == 'linear_points':
        plt.annotate('', xy=(215, 90), xytext=(210, 90),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
        plt.annotate('Real Wind', xy=(210, 91))
        plt.plot(data[0],data[1],data[0],data[1],'o')
        
def config_subplots(num,data,axisX,axisY,title,graph_type,x=0):
    dim = int(np.ceil(np.sqrt(num)))
    fig, axs = plt.subplots(dim,dim)
    fig.set_size_inches(10,7)
    if graph_type == 'points':
        x = np.array(range(np.size(data[0])))+1
    
    for i in range(dim):
        for j in range(dim):
            axs[i, j].set_title(title[dim*i+j])
            if graph_type == 'points':
                axs[i, j].plot(x, data[dim*i+j], 'o')
            elif graph_type == 'Vbar':
                axs[i, j].bar(x[dim*i+j],data[dim*i+j])
                for k in range(len(data[dim*i+j])):
                    axs[i, j].annotate(str(data[dim*i+j][k]),(k-0.3,data[dim*i+j][k]+1))         
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

path = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/results/data'
images = ['graph_times.png','histogram_times.png','Test_obj_points.png','Error_graphs.png',
          'Histogram_errors.png']
image = 4
num_experiments = 1024

test_files,train_files = files_list(num_experiments)
fail_test, not_fail_test = fail_list(test_files)
fail_train, not_fail_train = fail_list(train_files)
print('Total fail test scenarios: ', len(fail_test), ' %:', len(fail_test)*100/num_experiments)
print('Total not fail test scenarios: ', len(not_fail_test), ' %:', len(not_fail_test)*100/num_experiments)
print('Total fail train scenarios: ', len(fail_train), ' %:', len(fail_train)*100/num_experiments)
print('Total not fail train scenarios: ', len(not_fail_train), ' %:', len(not_fail_train)*100/num_experiments)

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
        x,y = config_bar_data(y,500,2500,20)
        config_plot([x,y], axisX, axisY, title, 'Vbar')
    elif image == 2:
        axisX = 'X'
        axisY = 'Y'
        title = 'Test objective points'
        obj_points = environment_points(True)
        x = np.array(obj_points[0])
        y = np.array(obj_points[1])
        config_plot([x,y], axisX, axisY, title, 'linear_points')
    elif image == 3:
        axisX = 'Number of designing point'
        axisY = 'MAPE'
        title = ['MAPE for navigation type 1','MAPE for navigation type 2',
                 'MAPE for navigation type 3','MAPE for navigation type 4']
        error = error_graphs(not_fail_test, path)
        error = np.transpose(error)
        fig = config_subplots(4,error,axisX,axisY,title,'points')
        fig.savefig(images[image])
    elif image == 4:
        inf = [2,0,0,0]
        sup = [16,10,6,4]
        step = [14,14,14,14]
        axisX = 'MAPE'
        axisY = 'Number of experiments'
        title = ['Number of test experiments for each MAPE (type 1)',
                 'Number of test experiments for each MAPE (type 2)',
                 'Number of test experiments for each MAPE (type 3)',
                 'Number of test experiments for each MAPE (type 4)']
        error = error_graphs(not_fail_test, path)
        error = np.transpose(error)
        x,y = config_bar_data(error[0],2,16,14)
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
    if image != 3 and image != 4:
        plt.savefig(images[image])
    
    
except:
    
    print('Error')