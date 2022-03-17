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
        
def config_bar_data(data,start,finish,intervals):
    limits = [start]
    x = []
    step = (finish - start)//intervals
    while limits[-1]<finish:
        limits.append(limits[-1]+step)
    y, xn = np.histogram(data, bins=limits)
    for i in range(len(xn)-1):
        x.append(str(xn[i]))
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


def error_data(test_files,path):
    data = []
    scenario = [1,1]
    file = test_files[0]
    f = open(path+'/'+file,'r')
    info = f.read().split('\n')
    for i in range(len(info)):
        split_info = info[i].split(',',1)
        scenario[1] = int(split_info[1][0])
        
            
            
    
def error_metric(data):
    y_mean = np.mean(data)
    MAE = 0
    for i in data:
        MAE += np.abs(i-y_mean)
    MAE /= len(data)
    return MAE

path = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/results/data'
images = ['graph_times.png','histogram_times.png','Test_obj_points.png']
image = 2
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
        axisX = 'X'
        axisY = 'Y'
        title = 'Test objective points'
        obj_points = environment_points(True)
        x = np.array(obj_points[0])
        y = np.array(obj_points[1])
        config_plot([x,y], axisX, axisY, title, 'linear_points')
        plt.annotate('Real Wind', xy=(213, 90), xytext=(210, 90),
                     arrowprops=dict(facecolor='black', shrink=0.0),
                     )
    plt.savefig(images[image])
    
    
except:
    
    print('Error')