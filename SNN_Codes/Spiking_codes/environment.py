#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:14:17 2021

@author: nelson
"""
import numpy as np
import copy as cp

class sailboat_environment:
    
    waypoints = []
    m=0
    theta=0
    b1=0
    b2=0
    state = 0
    controllers = []
    n_data = []
    distance = 0
    sensor_max = []
    sensor_min = []
    restart = 2
    rewards = [0,0]
    hyperparam = []
    saving = 0
    path = ''
    learning =  True
    prev_angle = 0
    sum_angle = 0
    actual_speed = 0
    tack = False
    tack_direction = -1
    desired_heading = 0
    
    
    def __init__(self, rudder_ctrl, sail_ctrl, vmax, vmin, hyperparam, path, learning = True):
        self.controllers.append(rudder_ctrl)
        self.controllers.append(sail_ctrl)
        self.sensor_max = vmax
        self.sensor_min = vmin
        self.hyperparam = hyperparam
        self.path = path
        #self.learning = learning
        self.train_scenario(train=False)
        
    def train_scenario(self, train=True):
        if train:
            center = []
            initial_point = [240, 100, 0]          
            center.append(self.hyperparam[9])
            center.append(self.hyperparam[10])
            r = np.sqrt((initial_point[0]-center[0])**2+(initial_point[1]-center[1])**2)
            n = self.hyperparam[11]
            theta=np.arctan2(initial_point[1]-center[1], initial_point[0]-center[0])
            
            for i in range(n-1):
                theta += 2*np.pi/n
                x = int(r*np.cos(theta)+center[0])
                y = int(r*np.sin(theta)+center[1])
                self.waypoints.append([x,y,0])
            np.random.shuffle(self.waypoints)
            self.waypoints.insert(0, initial_point)
            
        else:
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
            self.waypoints=[
                [240.0, 100.0, 0.0], 
                [275.0, 100.0, 0.0]
            ]
    def reward(self, data):
        
        for k in range(len(self.controllers)):
            if self.controllers[k].is_rudder_controller: 
                
                # objective = 180*np.arctan2(self.waypoints[self.state+1][1]-data[2],
                #                            self.waypoints[self.state+1][0]-data[1])/np.pi 
                heading = data[5]
                #actual_heading = data[5]
                #real_wind_angle = data[6] + actual_heading
                
                real_st = self.real_action(real_value=heading,
                                           desired_value = self.desired_heading,
                                           min_value = self.sensor_min[0],
                                           max_value = self.sensor_max[0],
                                           num_state = self.hyperparam[3])
                # if  self.tack:
                    
                    
                #     # to_no_go_zone = (alpha<=self.hyperparam[4]/2 and alpha>=-self.hyperparam[4]/2) or (180-self.hyperparam[4]*0.5<=alpha or 0.5*self.hyperparam[4]-180>=alpha)
                    
                #     # err_ang = objective - actual_heading
        
                #     # if to_no_go_zone and err_ang >0:
                #     #     objective = 180-self.hyperparam[5]
                #     # elif to_no_go_zone and err_ang <0:
                #     #     objective = -180+self.hyperparam[5]
                #     # elif not to_no_go_zone and err_ang >0:
                #     #     objective = -180+self.hyperparam[5]  #real_wind_angle+hyperparam[5]
                #     # else:
                #     #     objective = 180-self.hyperparam[5] 
                        
                #     # real_st = self.real_action(real_value=heading, 
                #     #                            desired_value = objective, 
                #     #                            min_value = self.sensor_min[0], 
                #     #                            max_value = self.sensor_max[0], 
                #     #                            num_state = self.hyperparam[3])
                    
                #     # # if to_no_go_zone and real_st <= self.hyperparam[3]//2:
                #     # #     desired_state=self.hyperparam[3]
                #     # # elif to_no_go_zone and real_st < self.hyperparam[3]//2:
                #     # #     desired_state=0
                #     # # elif not to_no_go_zone and real_st >= self.hyperparam[3]//2:
                #     # #     desired_state=0
                #     # # else:
                #     # #     desired_state=hyperparam[3]
                    
                #     # # r = self.puntual_reward(real_state=real_st, desired_state=self.hyperparam[3]//2, 
                #     # #                         num_states = self.hyperparam[3])
                #     print("Tacking")
                # else:
                    
                #     real_st = self.real_action(real_value=heading, 
                #                                desired_value = objective, 
                #                                min_value = self.sensor_min[0], 
                #                                max_value = self.sensor_max[0], 
                #                                num_state = self.hyperparam[3])
                
            else:
                              
                real_st = self.real_action(real_value=self.prev_angle, 
                       desired_value = self.prev_sail_objective, 
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
        
        self.m = (r[1]-ro[1])/(r[0]-ro[0])
        self.theta = np.arctan2(r[1]-ro[1], r[0]-ro[0])
        if(np.abs(self.theta)>np.pi/2):
            self.theta+=-np.pi*self.theta/np.abs(self.theta)
        self.b1 = r[1]-self.m*r[0]+0.5*(w+4*self.tack)*(np.cos(self.theta)+self.m*np.sin(self.theta))
        self.b2 = r[1]-self.m*r[0]-0.5*(w+4*self.tack)*(np.cos(self.theta)+self.m*np.sin(self.theta))
        
    def is_restart(self,r,control_action):
        rst=2
        if r[1]<=self.m*r[0]+self.b1 and r[1]>=self.m*r[0]+self.b2:
            rst=control_action
        
        self.restart = rst
    
    def get_plane(self,orig_x,orig_y,next_x,next_y):
        l=[]
        orig=[]        
        if self.restart == 2:
            self.state = 0           
        orig.append(orig_x)
        orig.append(orig_y)
        l.append(next_x)
        l.append(next_y)
    
        return orig,l
    
    def control_inputs(self, data, max_rate, min_rate):
        self.distance = np.sqrt((data[1]-self.waypoints[self.state+1][0])**2+(data[2]-self.waypoints[self.state+1][1])**2)
        self.n_data = []
        
        actual_heading = data[5] 
        real_wind_angle = data[6] + actual_heading
        self.actual_speed = np.sqrt(data[7]**2+data[8]**2)
        
        for i in range(len(self.controllers)):
            if self.controllers[i].is_rudder_controller: # State coding, this metod works with MSTDP, choosen method
                l = [min_rate]*int(3*self.hyperparam[0]*self.hyperparam[1]);
                pitch = data[3]
                alpha = -data[6] # A variable for detect if the sailboat is on tacking zone
                self.desired_heading = 180*np.arctan2(self.waypoints[self.state+1][1]-data[2],
                                                      self.waypoints[self.state+1][0]-data[1])/np.pi
                  # A variable for detect if the objective is on tacking zone
                beta = self.angle_saturation(ang = self.desired_heading - real_wind_angle, 
                                             min_ang=-180, 
                                             max_ang=180)
                
                to_tacking_zone_UW =  180-self.hyperparam[2]*0.5<=alpha or 0.5*self.hyperparam[2]-180>=alpha
                obj_is_tacking_UW =  180-self.hyperparam[2]*0.5<=beta or 0.5*self.hyperparam[2]-180>=beta 
                to_tacking_zone_TW =  alpha<=self.hyperparam[2]/2 and alpha>=-self.hyperparam[2]/2
                obj_is_tacking_TW = beta<=self.hyperparam[2]/2 and beta>=-self.hyperparam[2]/2
                
                if  to_tacking_zone_UW and obj_is_tacking_UW:                   
                    n3 = self.refresh_tack_direction(heading_angle = actual_heading, 
                                                     real_wind = real_wind_angle,
                                                     tack_angle=self.tack_direction*self.hyperparam[5])
                    self.desired_heading = self.tack_direction*self.hyperparam[5]+real_wind_angle
                    self.tack = True
                elif to_tacking_zone_TW and obj_is_tacking_TW:                   
                    n3 = self.refresh_tack_direction(heading_angle = actual_heading, 
                                                     real_wind = real_wind_angle,
                                                     tack_angle = self.tack_direction*self.hyperparam[13])
                    self.desired_heading = self.tack_direction*self.hyperparam[13]+real_wind_angle
                    self.tack = True
                else:
                    self.tack = False
                    n3 = 0
                    
                err_ang = self.angle_saturation(ang=self.desired_heading - actual_heading, 
                                                min_ang=-180, 
                                                max_ang=180)
                 
                if abs(err_ang)>=self.sensor_max[0]:
                    err_ang = (self.sensor_max[0]-1)*err_ang/abs(err_ang)
                if abs(pitch)>=self.sensor_max[2]:
                    pitch = (self.sensor_max[2]-1)*pitch/abs(pitch)
                
                n1 = ((err_ang+self.sensor_max[0])*self.hyperparam[0])//180
                n2 = ((pitch+self.sensor_max[2])*2*self.hyperparam[1])//180      
        
                l[int(self.hyperparam[0]*(self.hyperparam[1]*n3+n2)+n1)] = max_rate
                self.n_data.append(l)
                
            else:
                
                l = [min_rate]*int(3*self.hyperparam[7]);
                angle_apparent = self.aparent_wind(real_wind_angle = real_wind_angle, 
                                  sailboat_speed = self.actual_speed,
                                  yaw = actual_heading)
                self.sum_angle = self.angle_saturation(ang = angle_apparent + actual_heading, 
                                                       min_ang=-180, 
                                                       max_ang=180)
                n = (self.hyperparam[7]*(self.sum_angle+180))//360
                n2 = self.waypoints[self.state+1][2]
                l[int(self.hyperparam[7]*n2+n)] = max_rate
                self.n_data.append(l)

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
            
        return desired_state
    
    def calculate_reward(self, data):
           
        if self.restart ==2 or self.restart ==1:
            if not self.tack:
                ro,r = self.get_plane(orig_x = self.waypoints[self.state][0],
                                      orig_y = self.waypoints[self.state][1],
                                      next_x = self.waypoints[self.state+1][0],
                                      next_y = self.waypoints[self.state+1][1])
            else:
                ro,r = self.get_plane(orig_x = data[1],
                                      orig_y = data[2],
                                      next_x = data[1]+np.cos(np.radians(self.desired_heading)),
                                      next_y = data[2]+np.sin(np.radians(self.desired_heading)))
                
            self.carril_velero(ro,r,self.hyperparam[6])
            self.rewards = [0,0]
            for i in range(len(self.controllers)):
                self.controllers[i].reset_state_variables()
        else:        
            self.reward(data=data)
            
    def environment_step(self, data, max_rate, min_rate):
        
        control_action = [0,0,0]
        self.control_inputs(data = data, max_rate = max_rate, min_rate = min_rate)
        self.calculate_reward(data = data)
        for i in range(len(self.controllers)):
            control_action[i] = int(self.controllers[i].train_episode(n_data = self.n_data[i],
                                                                      reward = self.rewards[i]))
        control_action[2] = self.is_finish()   
        self.is_restart([data[1],data[2]], control_action[2])
        self.prev_angle = control_action[1]
        self.prev_sail_objective = self.sail_aproximation(prev_yaw=data[5])
        control_action[1] = self.prev_sail_objective
        control_action[2] = self.restart
        print(control_action[0])

        return control_action
            
    def is_finish(self):
        if self.distance<2:
            final = 1
        else:
            final = 0
        return final
    
    def save_SNN_state(self):
        if(self.restart==1):
            self.state += 1
        if self.saving < self.state:
            for i in range(len(self.controllers)):
                self.controllers[i].save_SNN(self.path)
            self.saving +=1
    
    def aparent_wind(self, real_wind_angle, sailboat_speed, yaw):
        
        i = sailboat_speed
        theta = np.arctan2(np.sin(np.pi*real_wind_angle/180)-i*np.sin(np.pi*yaw/180),
                           np.cos(real_wind_angle)-i*np.cos(yaw))
        return 180*theta/np.pi
    
    def sail_aproximation(self, prev_yaw):
        
        theta = [0,0]
        angle = self.sum_angle*np.pi/180
        theta_sail = np.arctan(np.sin(angle)/(np.cos(angle)-1))*180/np.pi
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
            
    def refresh_tack_direction(self,heading_angle,real_wind,tack_angle):
        
        h = self.angle_saturation(ang = heading_angle-real_wind, 
                                  min_ang=-180, 
                                  max_ang=180)
        
        l = self.angle_saturation(ang = self.desired_heading - real_wind, 
                                  min_ang=-180, 
                                  max_ang=180)
        if h == 0:
            h = 0.01
        if not self.tack:
            self.tack_direction = h/np.abs(h)
        elif self.actual_speed > self.hyperparam[12] and h<tack_angle+20 and h>tack_angle-20 and ((l<10 and l>-10) or l<-170 or l>170):
            self.restart = 1
            self.tack_direction *= -1
            
        return 0.5*(self.tack_direction+3)
    
    def angle_saturation(self,ang,min_ang,max_ang,degree=True):
        complete = (degree*360)+2*(1-degree)*np.pi
        if ang < min_ang:
            ang += complete
        elif ang >= max_ang:
            ang -= complete
        return ang