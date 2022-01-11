#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:36:50 2022

@author: nelson
"""
import numpy as np

class train_test_scenarios:
    
    hyperparam = []
    initial_point = [240, 100, 0] 
    waypoints = []
    learning =  True
    number_train = 0
    scenario = 0
    
    def train_scenario(self, test = False):
        self.waypoints = []
        
        if test:
            center = [0,0]
            center [0] = self.hyperparam[9]
            center [1] = self.hyperparam[10]
            r = np.sqrt((self.initial_point[0]-center[0])**2+(self.initial_point[1]-center[1])**2)
            n = self.hyperparam[11]
            theta=np.arctan2(self.initial_point[1]-center[1], self.initial_point[0]-center[0])
            
            for i in range(n-1):
                theta += 2*np.pi/n
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,0])
            
        elif self.scenario == 0:
            center = self.initial_point
            n = self.hyperparam[14]
            theta=np.pi
            r = 40
            
            for i in range(n):           
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,0])
                theta -= np.pi/(n-1)
                
            # self.waypoints=[
            #     [240.0, 100.0, 0.0],
            #     [255.0, 95.0, 0.0], #(255.0, 100.0, 0.0)
            #     [260.0, 105.0, 0.0], #(260.0, 105.0, 0.0)
            #     [265.0, 100.0, 0.0], #(265.0, 100.0, 0.0)
            #     [270.0, 95.0, 0.0], #(270.0, 95.0, 0.0)
            #     [275.0, 100.0, 0.0],
            #     [270.0, 105.0, 0.0],
            #     [265.0, 100.0, 0.0],
            #     [260.0, 95.0, 0.0],
            #     [255.0, 100.0, 0.0],
            #     [250.0, 105.0, 0.0],
            #     [245.0, 100.0, 0.0],
            #     [240.0, 100.0, 0.0]
            # ]
            # self.waypoints=[
            #     [240.0, 100.0, 0.0],
            #     #[270.0, 95.0, 0.0]
            #     [268.0, 67.0, 0.0]
            # ]
        elif self.scenario == 1:
            center = self.initial_point
            n = self.hyperparam[14]
            theta=-np.pi
            r = 40
            
            for i in range(n):           
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,1])
                theta += np.pi/(n-1)
                
            
        else:
            center = self.initial_point # Turn the wind to
            n = self.hyperparam[16]
            theta=np.pi
            r = 50
            
            for i in range(n):        
                
                theta -= np.pi/(n+1)
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,0])
            
        self.number_train = n
        np.random.shuffle(self.waypoints)
        self.waypoints.insert(0, self.initial_point)