# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:50:03 2022

@author: Usuario
"""
import numpy as np
import matplotlib.pyplot as plt
import oapackage as OA

class Pareto_optimization:
    maximization = True  
    
    def __init__(self, maximization = True):
        self.maximization = maximization
        
    def minimization_problem(self,data):
        datapoints = data
        i = 0
        if not self.maximization:
            for row in data:
                max_point = row.max()
                min_point = row.min()
                m = (max_point-min_point)/(min_point-max_point)
                b = max_point - m*min_point
                datapoints[i] = m*row + b
                i+=1
        return datapoints
    
    def resize_info(self, datapoints):
        inp = []
        for row in range(len(datapoints[0])):
            t = []
            for col in range(len(datapoints)):
                t.append(datapoints[col][row])
            inp.append(t)
        return inp
    
    def example_pareto(self):       
        datapoints=np.random.rand(3, 50)
        datapoints[0]*=10

        plt.plot(datapoints[0,:], datapoints[1,:], '.b', markersize=16, label='Non Pareto-optimal')
        _=plt.title('The input data', fontsize=15)
        plt.xlabel('Objective 1', fontsize=16)
        plt.ylabel('Objective 2', fontsize=16)
        
        lst=self.pareto_optimum(datapoints) # the indices of the Pareto optimal designs
        
        optimal_datapoints=datapoints[:,lst]
        
        plt.plot(datapoints[0,:], datapoints[1,:], '.b', markersize=16, label='Non Pareto-optimal')
        plt.plot(optimal_datapoints[0,:], optimal_datapoints[1,:], '.r', markersize=16, label='Pareto optimal')
        plt.xlabel('Objective 1', fontsize=16)
        plt.ylabel('Objective 2', fontsize=16)
        plt.xticks([])
        plt.yticks([])
        _=plt.legend(loc=3, numpoints=1)
    
    def pareto_optimum(self,data):
        datapoints = self.minimization_problem(data)
        pareto=OA.oapackage.ParetoDoubleLong()

        inp = self.resize_info(datapoints)
        for ii in range(0, datapoints.shape[1]):
            w=OA.oapackage.doubleVector(inp[ii])
            pareto.addvalue(w, ii)
            
        datapoints = self.minimization_problem(datapoints)
        lst=pareto.allindices() # the indices of the Pareto optimal designs
        
        return lst