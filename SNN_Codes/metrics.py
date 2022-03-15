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
    if graph_type == 'points':
        plt.title(title)
        plt.xlabel(axisX)
        plt.ylabel(axisY)
        plt.plot(data[0],data[1],'o')
    elif graph_type == 'Vbar':
        plt.bar(data[0],data[1])
        for i in range(len(data[1])):
            plt.annotate(str(data[1][i]),(i-0.3,data[1][i]+1))
        plt.title(title)
        plt.xlabel(axisX)
        plt.ylabel(axisY)
        
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

path = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/results/data'
images = ['graph_times.png','histogram_times.png']
image = 1
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
    plt.savefig(images[image])
    
    
except:
    
    print('Error')