#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:14:17 2021

@author: nelson
"""
import numpy as np
import copy as cp
import train_scenarios as ts

class sailboat_environment(ts.train_test_scenarios):
    
    m=0
    theta=0
    b1=0
    b2=0
    state = 0
    n_data = []
    distance = 0
    sensor_max = []
    sensor_min = []
    restart = 2
    rewards = [0,0]
    saving = 0
    path = ''
    prev_angle = 0
    sum_angle = 0
    actual_speed = 0
    tack = False
    tack_direction = -1
    desired_heading = 0
    tack_sign = False
    tack_angle_logic = [-1000,-1000]
    tack_rst = False
    
    def __init__(self, rudder_ctrl, sail_ctrl, vmax, vmin, hyperparam, path, learning = True):
        self.controllers.append(rudder_ctrl)
        self.controllers.append(sail_ctrl)
        self.sensor_max = vmax
        self.sensor_min = vmin
        self.hyperparam = hyperparam
        self.path = path
        self.learning = learning
        self.train_scenario(test = not learning)      

    def reward(self, data):
        
        for k in range(len(self.controllers)):
            if self.controllers[k].is_rudder_controller:            
                heading = data[5]
                real_st = self.real_action(real_value=heading,
                                           desired_value = self.desired_heading,
                                           min_value = self.sensor_min[0],
                                           max_value = self.sensor_max[0],
                                           num_state = self.hyperparam[3])            
            else:
                              
                real_st = self.real_action(real_value=self.prev_sail_objective, 
                                           desired_value = self.prev_angle, 
                                           min_value = self.sensor_min[1], 
                                           max_value = self.sensor_max[1], 
                                           num_state = self.hyperparam[8])
                
            self.rewards[k] = self.puntual_reward(real_state=real_st, desired_state=self.hyperparam[3+5*k]//2, 
                                                  num_states = self.hyperparam[3+5*k])
            
    def normalize(self,data,vmax,vmin, A=1, B=0):
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
    
    def carril_velero(self,ro,r,w):
        
        const = 8
        x = r[0]-ro[0]
        if x==0:
            x=0.001
        self.m = (r[1]-ro[1])/x
        self.theta = np.arctan((r[1]-ro[1])/x)
        k = np.abs(0.5*(w+const*self.tack)/np.cos(self.theta))
        self.b1 = r[1]-self.m*r[0]+k
        self.b2 = r[1]-self.m*r[0]-k
        print (self.m,self.b1,self.b2)
        
    def is_restart(self,r,control_action):
        if r[1]<=self.m*r[0]+self.b1 and r[1]>=self.m*r[0]+self.b2:
            rst=control_action
        else:
            rst = 2
            self.tack = False
            self.tack_sign = False
        
        self.restart = rst
    
    def get_plane(self,orig_x,orig_y,next_x,next_y):
        l=[]
        orig=[]               
        orig.append(orig_x)
        orig.append(orig_y)
        l.append(next_x)
        l.append(next_y)
    
        return orig,l
    

    def puntual_reward(self,real_state, desired_state, num_states):
        
        cons = 1/(num_states-1)
        r = cons*(desired_state - real_state)
        
        if abs(r) > 1:
            r=abs(r)/r
        
        return r
    
    def real_action(self,real_value, desired_value, min_value, max_value, num_state):
        max_error = max_value - min_value
        size_error = 2*max_error
        error = desired_value - real_value
        desired_state = ((error + max_error)*num_state)//size_error
        
        if desired_state >= num_state-1:
            desired_state = num_state-1      
        elif desired_state < 0:
            desired_state = 0
            
        return desired_state
    
    def calculate_reward(self, data):
           
        if self.restart ==2 or self.restart ==1 or self.tack_rst:
            if not self.tack:
                ro,r = self.get_plane(orig_x = self.waypoints[0][0],
                                      orig_y = self.waypoints[0][1],
                                      next_x = self.waypoints[self.state+1][0],
                                      next_y = self.waypoints[self.state+1][1])
            else:
                ro,r = self.get_plane(orig_x = data[1],
                                      orig_y = data[2],
                                      next_x = data[1]+int(100*np.cos(np.radians(self.desired_heading))),
                                      next_y = data[2]+int(100*np.sin(np.radians(self.desired_heading))))
                
            self.carril_velero(ro,r,self.hyperparam[6])
            self.rewards = [0,0]
            self.tack_rst = False
            for i in range(len(self.controllers)):
                self.controllers[i].reset_state_variables()
        else:        
            self.reward(data=data)
            
    def is_finish(self):
        if self.distance<self.min_distance:
            final = 1
        else:
            final = 0
        return final
    
    def save_SNN_state(self):
        if(self.restart==1):
            self.state += 1
        # if (self.saving < self.state and not self.tack) or (self.tack and self.restart==1):
        #     for i in range(len(self.controllers)):
        #         self.controllers[i].save_SNN(self.path)
        #     self.saving +=1
        if self.state == self.number_train:
            self.controllers[1].network_name += str(self.scenario)
            for i in range(len(self.controllers)):
                self.controllers[i].save_SNN(self.path)
            self.state = 0
            self.scenario += 1
            self.train_scenario()
            
            
    def aparent_wind(self, real_wind_angle, sailboat_speed, yaw):
        
        i = sailboat_speed/np.sqrt(9)
        theta = np.arctan2(np.sin(np.pi*real_wind_angle/180)-i*np.sin(np.pi*yaw/180),
                           np.cos(real_wind_angle)-i*np.cos(yaw))
        
        return 180*theta/np.pi
    
    def sail_aproximation(self, prev_yaw):
        
        theta = [0,0]
        angle = self.sum_angle*np.pi/180
        theta_sail = np.arctan(np.sin(angle)/(np.cos(angle)-1))*180/np.pi
        if self.scenario == 2:
            theta_sail = np.arctan(np.sin(angle)/(np.cos(angle)+1))*180/np.pi
        if self.sum_angle == 0:
            theta_sail = 90
        theta[0] = theta_sail - prev_yaw
        theta[1] = theta[0] + 180
        opt_theta = 0
        for theta_sail in theta:
            alpha = self.angle_saturation(ang=theta_sail,
                                          min_ang=-180,
                                          max_ang=180)
            if alpha <= self.controllers[1].max_out and alpha >= self.controllers[1].min_out:
                  opt_theta = cp.copy(alpha)
        
        return opt_theta
            
    def refresh_tack_direction(self,heading_angle,real_wind,tack_angle,phase):
        h = self.angle_saturation(ang = heading_angle-real_wind,
                                  min_ang=-180,
                                  max_ang=180)
        v =  self.tack_angle_logic [0] != -1000 #Reset at change of desired point
        self.tack_angle_logic [1] = self.angle_saturation(ang = self.desired_heading - real_wind,
                                                          min_ang=-180,
                                                          max_ang=180)

        l = (self.tack_angle_logic [1] > 0 and self.tack_angle_logic [0] <= 0) or (self.tack_angle_logic [1] < 0 and self.tack_angle_logic [0] >= 0)
        if l and v:
            self.tack_sign = True
        if h == 0:
            h = 0.01
            
        l = self.angle_saturation(ang = h+phase,
                                  min_ang=-180,
                                  max_ang=180)
        if not self.tack:
            self.tack_direction = h/np.abs(h)
            self.tack_sign = (l>=0 and self.tack_angle_logic [1] >= 0) or (l<0 and self.tack_angle_logic [1] < 0)
        elif self.actual_speed > self.hyperparam[12] and l<tack_angle+30 and l>tack_angle-30 and self.tack_sign:
            self.tack_direction *= -1
            self.tack_sign = False
            self.tack_rst = True
            
        self.tack_angle_logic[0] = self.tack_angle_logic[1]
        return 0.5*(self.tack_direction+3)
    
    def angle_saturation(self,ang,min_ang,max_ang,degree=True):
        complete = (degree*360)+2*(1-degree)*np.pi
        while ang < min_ang or ang >= max_ang:
            if ang < min_ang:
                ang += complete
            else:
                ang -= complete
        return ang

    def environment_step(self, data, max_rate, min_rate):
        
        control_action = [0,0,0,0]
        self.save_SNN_state()
        self.control_inputs(data = data, max_rate = max_rate, min_rate = min_rate)
        self.calculate_reward(data = data)
        for i in range(len(self.controllers)):
            control_action[i] = int(self.controllers[i].train_episode(n_data = self.n_data[i],
                                                                      reward = self.rewards[i]))
        control_action[3] = self.is_finish()
        self.is_restart([data[1],data[2]], control_action[3])
        self.prev_angle = control_action[1]
        self.prev_sail_objective = self.sail_aproximation(prev_yaw=data[5])
        #control_action[1] = self.prev_sail_objective
        control_action[2] = control_action[1]
        control_action[3] = cp.copy(self.restart)
        if self.restart == 1 or self.restart == 2:
            control_action[3] = 2
        return control_action  
    
    def environment_test(self, data, max_rate, min_rate):
        control_action = [0,0,0,0]
        if(self.restart==1):
            self.state += 1
            self.tack_angle_logic [0] = -1000
            self.tack_angle_logic [1] = -1000
            self.tack = False
        self.control_inputs(data = data, max_rate = max_rate, min_rate = min_rate)
        for i in range(len(self.controllers)):
            control_action[i] = int(self.controllers[i].train_episode(n_data = self.n_data[i],
                                                                      reward = self.rewards[i]))
        control_action[3] = self.is_finish()
        self.restart = cp.copy(control_action[3])
        control_action[2] = control_action[1]
        print(self.waypoints[self.state+1])
        return control_action
    
    def control_inputs(self, data, max_rate, min_rate):
        self.distance = np.sqrt((data[1]-self.waypoints[self.state+1][0])**2+(data[2]-self.waypoints[self.state+1][1])**2)
        self.n_data = []
        
        actual_heading = data[5] 
        real_wind_angle = self.angle_saturation(ang = data[6] + actual_heading, 
                                                min_ang=-180, 
                                                max_ang=180)
        self.actual_speed = np.sqrt(data[7]**2+data[8]**2)
        print(self.actual_speed)
        for i in range(len(self.controllers)):
            if self.controllers[i].is_rudder_controller: # State coding, this metod works with MSTDP, choosen method
                l = [min_rate]*int(self.hyperparam[0]*self.hyperparam[1]);
                pitch = data[3]
                alpha = -data[6] # A variable for detect if the sailboat is on tacking zone
                self.desired_heading = 180*np.arctan2(self.waypoints[self.state+1][1]-data[2],
                                                      self.waypoints[self.state+1][0]-data[1])/np.pi
                  # A variable for detect if the objective is on tacking zone
                beta = self.angle_saturation(ang = self.desired_heading - real_wind_angle, 
                                             min_ang=-180, 
                                             max_ang=180)
                
                to_tacking_zone_UW =  180-self.hyperparam[2]*0.5<alpha or 0.5*self.hyperparam[2]-180>alpha
                obj_is_tacking_UW =  180-self.hyperparam[2]*0.5<beta or 0.5*self.hyperparam[2]-180>beta 
                to_tacking_zone_TW =  alpha<self.hyperparam[4]/2 and alpha>-self.hyperparam[4]/2
                obj_is_tacking_TW = beta<self.hyperparam[4]/2 and beta>-self.hyperparam[4]/2
                
                if  to_tacking_zone_UW and obj_is_tacking_UW:                   
                    n3 = self.refresh_tack_direction(heading_angle = actual_heading, 
                                                     real_wind = real_wind_angle,
                                                     tack_angle=-self.tack_direction*self.hyperparam[5],
                                                     phase = 180)
                    
                    self.desired_heading = self.angle_saturation(ang = -self.tack_direction*self.hyperparam[5]+real_wind_angle+180, 
                                                                 min_ang=-180, 
                                                                 max_ang=180)

                    self.tack = True
                elif to_tacking_zone_TW and obj_is_tacking_TW:                   
                    n3 = self.refresh_tack_direction(heading_angle = actual_heading, 
                                                     real_wind = real_wind_angle,
                                                     tack_angle = self.tack_direction*self.hyperparam[13],
                                                     phase = 0)
                    self.desired_heading = self.angle_saturation(ang = self.tack_direction*self.hyperparam[13]+real_wind_angle, 
                                                                 min_ang=-180, 
                                                                 max_ang=180)
                    self.tack = True
                else:
                    self.tack = False
                    n3 = 0
                
                if self.desired_heading != 0 and self.waypoints[self.state+1][2] == 1:
                    self.desired_heading -= 180*self.desired_heading/np.abs(self.desired_heading)
                elif self.desired_heading == 0:
                    self.desired_heading = 180
                err_ang = (1-2*self.waypoints[self.state+1][2]) * self.angle_saturation(
                                                                   ang=self.desired_heading - actual_heading, 
                                                                   min_ang=-180,
                                                                   max_ang=180)
                n3 = 0
                if abs(err_ang)>=self.sensor_max[0]:
                    err_ang = (self.sensor_max[0]-1)*err_ang/abs(err_ang)
                if abs(pitch)>=self.sensor_max[2]:
                    pitch = (self.sensor_max[2]-1)*pitch/abs(pitch)
                
                n1 = ((err_ang+self.sensor_max[0])*self.hyperparam[0])//180
                n2 = ((pitch+self.sensor_max[2])*2*self.hyperparam[1])//180      
        
                l[int(self.hyperparam[0]*(self.hyperparam[1]*n3+n2)+n1)] = max_rate
                self.n_data.append(l)
                
            else:
                
                l = [min_rate]*int(2*self.hyperparam[7]);
                angle_apparent = self.aparent_wind(real_wind_angle = real_wind_angle, 
                                  sailboat_speed = self.actual_speed,
                                  yaw = actual_heading)
                self.sum_angle = self.angle_saturation(ang = angle_apparent + actual_heading, #Is the apparent wind, no the real
                                                       min_ang=-180, 
                                                       max_ang=180)
                n = (self.hyperparam[7]*(self.sum_angle+180))//360
                n2 = self.waypoints[self.state+1][2]
                l[int(self.hyperparam[7]*n2+n)] = max_rate
                self.n_data.append(l)