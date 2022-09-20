#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:36:50 2022

@author: nelson
"""
import numpy as np

class train_test_scenarios:
    
    hyperparam = []
    initial_point = [6.267409626346163, -75.56918607339368, 0] 
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
            self.min_distance = 2
            self.hyperparam[2] = 90
            self.hyperparam[4] = 30
            self.hyperparam[5] = 45
            
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
            self.waypoints=[
                 [6.26744561974752, -75.56939327403767, 0.0],
                 [6.26728845280639, -75.56939289004683, 0.0]
            ]

        elif self.scenario == 0 or self.scenario == 2: #Direct
            center = self.initial_point
            n = self.hyperparam[14+(self.scenario == 2)]
            theta = np.pi/2
            r = 20
            self.previous[0] = self.hyperparam[2]
            self.previous[1] = self.hyperparam[4] 
            self.hyperparam[2] = 0
            self.hyperparam[4] = 0
            if self.scenario ==2:
                self.hyperparam[6] += 2
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
            r = 25
            #self.controllers[0].spiking_controller.learning = False
            self.hyperparam[2] = 180
            self.hyperparam[4] = 0
            self.min_distance = 4
            
            for i in range(n):        
                
                theta += np.pi/(n+1)
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,0])
                
        if not test:
            np.random.shuffle(self.waypoints)
        self.waypoints.insert(0, self.initial_point)
