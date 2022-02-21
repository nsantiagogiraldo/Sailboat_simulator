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
    controllers = []
    min_distance = 2
    previous = [0,0]
    
    def train_scenario(self, test = False):
        self.waypoints = []
        
        if test:
            band = True
            center = [0,0]
            if self.hyperparam[4] == 0:
                self.hyperparam[2] = self.previous[0]
                self.hyperparam[4] = self.previous[1]
            phi = 360/self.hyperparam[11]
            x = self.hyperparam[9]
            center[0] =  self.initial_point[0]-x
            center[1] =  self.initial_point[1]
            r = np.sqrt((self.initial_point[0]-center[0])**2+(self.initial_point[1]-center[1])**2)         
            
            n = 2*self.hyperparam[11]
            theta = 0
            
            for i in range(n-1):
                if band and i>= self.hyperparam[11]:
                    band = False
                    theta = 180 
                    center[0] = self.initial_point[0]+x
                    phi *= -1                   
                    
                theta += phi
                xf = int(r*np.cos(np.radians(theta))+center[0])
                yf = int(r*np.sin(np.radians(theta))+center[1])
                self.waypoints.append([xf,yf,0])
            
            # self.waypoints=[
            #     [240.0, 100.0, 0],
            #     [255.0, 95.0, 0], #(255.0, 100.0, 0.0)
            #     [260.0, 105.0, 0], #(260.0, 105.0, 0.0)
            #     [265.0, 100.0, 0], #(265.0, 100.0, 0.0)
            #     [270.0, 95.0, 0], #(270.0, 95.0, 0.0)
            #     [275.0, 100.0, 0],
            #     [270.0, 105.0, 0],
            #     [265.0, 100.0, 0],
            #     [260.0, 95.0, 0],
            #     [255.0, 100.0, 0],
            #     [250.0, 105.0, 0],
            #     [245.0, 100.0, 0],
            #     [240.0, 100.0, 0]
            # ]
            
            # self.waypoints=[
            #      [240.0, 100.0, 0],
            #      [252.0, 110.0, 0],
            #      [264.0, 90.0, 0],
            #      [276.0, 100.0, 0],
            #      [240.0, 100.0, 0]
            #  ]
            # self.waypoints=[
            #     [240.0, 105.0, 0.0],
            #     #[270.0, 95.0, 0.0]
            #     [270.0, 100.0, 0.0]
            # ]

        elif self.scenario == 0 or self.scenario == 2: #Direct
            center = self.initial_point
            n = self.hyperparam[14+(self.scenario == 2)]
            theta = np.pi/2
            r = 15
            self.previous[0] = self.hyperparam[2]
            self.previous[1] = self.hyperparam[4] 
            self.hyperparam[2] = 0
            self.hyperparam[4] = 0
            if self.scenario ==2:
                self.hyperparam[6] += 4
                self.controllers[0].spiking_controller.learning = False
            
            for i in range(n):           
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,int(self.scenario == 2)])
                theta -= np.pi/(n-1)
 
        elif self.scenario == 1:
            center = self.initial_point # Turn the wind. Tacking
            n = self.hyperparam[16]
            theta=np.pi/2
            r = 20
            #self.controllers[0].spiking_controller.learning = False
            self.hyperparam[2] = 180
            self.hyperparam[4] = 0
            self.min_distance = 4
            
            for i in range(n):        
                
                theta += np.pi/(n+1)
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,0])
                
        self.number_train = n
        if not test:
            np.random.shuffle(self.waypoints)
        self.waypoints.insert(0, self.initial_point)
        self.waypoints.insert(len(self.waypoints), self.initial_point)
