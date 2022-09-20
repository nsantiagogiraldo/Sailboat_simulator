#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:14:17 2021

@author: nelson
"""
import numpy as np
import copy as cp
import train_scenarios as ts
import text_file as db

class sailboat_environment(ts.train_test_scenarios):
    
    m = 0
    theta = 0
    b1 = 0
    b2 = 0
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
    file_number = 1
    prev_scenario = 0
    base = ''
    learning = True
    epoch = 1
    prev_rudder = 0
    heading = 0
    err_ang = 0
    real_wind_angle = 0
    ah = 0
    
    def __init__(self, rudder_ctrl, sail_ctrl, vmax, vmin, hyperparam, path, learning = True):
        self.controllers.append(rudder_ctrl)
        self.controllers.append(sail_ctrl)
        self.sensor_max = vmax
        self.sensor_min = vmin
        self.hyperparam = hyperparam
        self.path = path
        self.learning = learning
        self.epoch = 1
        self.train_scenario(test = not learning)     
         
    def dist_GPS(self,lati,longi,R=6378000):
        lon1 = np.radians(longi)
        lat1 = np.radians(lati)
        h=np.sin((lat1[0]-lat1[1])/2)**2+np.cos(lat1[0])*np.cos(lat1[1])*np.sin((lon1[0]-lon1[1])/2)**2
        d=2*R*np.arcsin(np.sqrt(h))
        return d
	
    def true_wind(self,app_wind_angle,app_wind_speed,sailboat_speed):
        if app_wind_speed==0:
            app_wind_speed=0.001
        w = np.sqrt(app_wind_speed**2+sailboat_speed**2-2*app_wind_speed*sailboat_speed*np.cos(app_wind_angle))
        true_wind_attack = np.arccos((app_wind_speed*np.cos(app_wind_angle)-sailboat_speed)/w)
        return w,true_wind_attack
        
    def angle_vectors(self,vect1,vect2):
        m_vec1=np.linalg.norm(vect1)
        m_vec2=np.linalg.norm(vect2)
        angle = np.arccos(np.dot(vect1,vect2)/(m_vec1*m_vec2))
        if vect1[0]<0:
            angle *= -1
        return np.degrees(angle)
           
    def set_database(self,db_name,path,structure):
        self.base = db.text_files(new_file = True, file_name = db_name, path = path, structure = structure) 
        
    def save_data(self,data,control_action = 0):

        if self.learning:
            
            self.base.append_data([self.scenario,self.state+1,self.rewards[0], 
                                   self.rewards[1],self.epoch,self.real_wind_angle,
                                   data[1],data[2],self.actual_speed])
            
        else:
            
            self.base.append_data([self.state+1,data[6],
                                   data[1],data[2],self.actual_speed,self.ah,
                                   control_action[0], control_action[1],data[3]])
        
    def reward(self, data):
        
        for k in range(len(self.controllers)):
            if self.controllers[k].is_rudder_controller:            
                real_st = self.real_action(real_value= -self.heading,
                                            desired_value = self.prev_rudder,
                                            min_value = self.sensor_min[0],
                                            max_value = self.sensor_max[0],
                                            num_state = 2*self.hyperparam[3]-2)            
                # real_st = self.real_action(real_value= data[5],
                #                            desired_value = self.desired_heading,
                #                            min_value = self.sensor_min[0],
                #                            max_value = self.sensor_max[0],
                #                            num_state = 2*self.hyperparam[3]-2) 
                                
            else:
                              
                real_st = self.real_action(real_value=self.prev_sail_objective, 
                                           desired_value = self.prev_angle, 
                                           min_value = self.sensor_min[1], 
                                           max_value = self.sensor_max[1], 
                                           num_state = 2*self.hyperparam[8]-2)
                
            self.rewards[k] = self.puntual_reward(real_state=real_st, desired_state=(self.hyperparam[3+5*k]-1), 
                                                  num_states = 2*self.hyperparam[3+5*k]-2)
            
            
        self.rewards[0]*=3
        #self.rewards[0]*=0.6
        #print(self.heading,real_st,self.rewards[0])

            
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
        error = self.angle_saturation(ang = desired_value - real_value,
                                      min_ang=-180,
                                      max_ang=180)
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
            final = self.file_number + 2
        return final
    
    def save_SNN_state(self):
        band = True
        if(self.restart==1):
            self.state += 1
        if self.state == self.number_train:
            for i in range(len(self.controllers)):
                self.controllers[i].save_SNN(self.path)
            self.scenario += 1
            if self.scenario!=3:
                self.train_scenario()
            self.state = 0
            band = False
        return band
            
            
    def aparent_wind(self, real_wind_angle, sailboat_speed, yaw):
        
        i = sailboat_speed/(1*2.2)
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
        self.control_inputs(data = data, max_rate = max_rate, min_rate = min_rate)
        self.calculate_reward(data = data)
        self.heading = cp.copy(self.err_ang)
        for i in range(len(self.controllers)):
            control_action[i] = int(self.controllers[i].train_episode(n_data = self.n_data[i],
                                                                      reward = self.rewards[i]))
        control_action[3] = self.is_finish()
        self.is_restart([data[1],data[2]], control_action[3])
        self.prev_angle = control_action[1]
        self.prev_sail_objective = self.sail_aproximation(prev_yaw=data[5])
        self.prev_rudder = control_action[0]
        
        #control_action[1] = self.prev_sail_objective
        control_action[2] = control_action[1]
        control_action[3] = cp.copy(self.restart)
        act_state = self.save_SNN_state()
        if (self.restart == 1 or self.restart == 2) and act_state:
            control_action[3] = 2
            self.epoch += 1
        elif not act_state:
            control_action[3] = 200
            self.prev_scenario += 1
        self.save_data(data)

        return control_action  
    
    def environment_test(self, data, max_rate, min_rate):
        control_action = [0,0,0,0]
        self.control_inputs(data = data, max_rate = max_rate, min_rate = min_rate)
        for i in range(len(self.controllers)):
            control_action[i] = int(self.controllers[i].train_episode(n_data = self.n_data[i],
                                                                      reward = 0))
        control_action[3] = self.is_finish()
        self.restart = cp.copy(control_action[3])
        control_action[2] = control_action[1]
        self.save_data(data,control_action)
        if(self.restart==1):
            self.state += 1
            self.tack_angle_logic [0] = -1000
            self.tack_angle_logic [1] = -1000
            self.tack = False
        # control_action[1] = self.sail_aproximation(prev_yaw=data[5])
        # control_action[2] = control_action[1]
        return control_action
    
    def environment_PI_test(self, port):
        data=port.read_data_sensor()
        data2 = self.control_inputs_PI(data)
        if not isinstance(data2,bool):
            control = port.classic_control_action(data2)
            control_action = self.is_finish()
            self.restart = cp.copy(control_action)
            port.write_control_action(control)
            self.save_data(data,control)
            if(self.restart==1):
                self.state += 1
                self.tack_angle_logic [0] = -1000
                self.tack_angle_logic [1] = -1000
                self.tack = False
        else:
            print("No hay dato")
    
    def environment_Viel2019_test(self, data):
        control_action = [0,0,0,0]
        actions = self.control_inputs_Viel(data = data)
        c = self.Viel2019(delta_rmax = np.radians(45), 
                          delta_smax = np.radians(90), 
                          d_theta = actions[0], 
                          theta = actions[1], 
                          phi = actions[2], 
                          sigma_awind = actions[3], 
                          Vi_asterisc = 1, 
                          epsilon = 0)
        control_action[0] = c[0]
        control_action[1] = c[1]
        control_action[3] = self.is_finish()
        self.restart = cp.copy(control_action[3])
        control_action[2] = control_action[1]
        self.save_data(data,control_action)
        if(self.restart==1):
            self.state += 1
            self.tack_angle_logic [0] = -1000
            self.tack_angle_logic [1] = -1000
            self.tack = False
        return control_action
        
    def environment_test_real(self, data, max_rate, min_rate):
        control_action = [0,0,0,0]
        self.control_inputs(data = data, max_rate = max_rate, min_rate = min_rate, real = True)
        for i in range(len(self.controllers)):
            control_action[i] = int(self.controllers[i].train_episode(n_data = self.n_data[i],
                                                                      reward = 0))
        control_action[3] = self.is_finish()
        self.restart = cp.copy(control_action[3])
        control_action[2] = control_action[1]
        self.save_data(data,control_action)
        if(self.restart==1):
            self.state += 1
            self.tack_angle_logic [0] = -1000
            self.tack_angle_logic [1] = -1000
            self.tack = False
        return control_action
            
    def Viel2019(self,delta_rmax,delta_smax,d_theta,theta,phi,sigma_awind,Vi_asterisc,epsilon):
    #The desired acceleration Vi* > 0, for all experiments
        if np.cos(theta-phi) - np.cos(epsilon) >= 0:
            Theta = phi
        else:
            Theta = theta
        if np.cos(Theta-d_theta) >= 0:
            delta_r = delta_rmax*np.sin(Theta-d_theta)
        else:
            delta_r = delta_rmax*np.sign(np.sin(Theta-d_theta))
       
        delta_s_opt = 0.25*np.pi*(np.cos(sigma_awind)+1)
        delta_sm = min([np.abs(np.pi-np.abs(sigma_awind)),delta_smax])
        if Vi_asterisc>0:
            delta_s = -np.sign(sigma_awind)*min([np.abs(delta_s_opt),delta_sm])
        elif Vi_asterisc<0:
            delta_s = -np.sign(sigma_awind)*delta_sm
   
        return np.degrees([delta_r,delta_s])
    
    def control_inputs(self, data, max_rate, min_rate, real = False):
    
        if real:
            magnetric_desv = -7
            self.actual_speed = data[7]
            actual_heading = self.angle_saturation(ang = data[5] + magnetric_desv, min_ang=-180, max_ang=180)
            self.ah=actual_heading
            self.distance = self.dist_GPS([data[1],self.waypoints[self.state+1][0]],[data[2],self.waypoints[self.state+1][1]])
            speed_wind,attack_angle = self.true_wind(data[6],data[3],self.actual_speed)
            self.real_wind_angle = self.angle_saturation(ang = attack_angle + actual_heading, min_ang=-180, max_ang=180)
            dvec=[self.waypoints[self.state+1][0]-data[1],self.waypoints[self.state+1][1]-data[2]]
            self.desired_heading = self.angle_vectors(dvec,np.array([1,0]))
            angle_apparent = self.angle_saturation(ang = data[6] + actual_heading, min_ang=-180, max_ang=180)
        else:
            self.distance = np.sqrt((data[1]-self.waypoints[self.state+1][0])**2+(data[2]-self.waypoints[self.state+1][1])**2)      
            actual_heading = data[5] 
            attack_angle = data[6]
            self.real_wind_angle = self.angle_saturation(ang = data[6] + actual_heading, 
                                                    min_ang=-180, 
                                                    max_ang=180)
            self.actual_speed = np.sqrt(data[7]**2+data[8]**2)        
            self.desired_heading = 180*np.arctan2(self.waypoints[self.state+1][1]-data[2],self.waypoints[self.state+1][0]-data[1])/np.pi
            angle_apparent = self.aparent_wind(real_wind_angle = self.real_wind_angle, sailboat_speed = self.actual_speed, yaw = actual_heading)
            
        self.n_data = []
        for i in range(len(self.controllers)):
            if self.controllers[i].is_rudder_controller: # State coding, this metod works with MSTDP, choosen method
                l = [min_rate]*int(self.hyperparam[0]*self.hyperparam[1]);
                #pitch = data[4]
                pitch = -attack_angle
                alpha = -attack_angle # A variable for detect if the sailboat is on tacking zone
                  # A variable for detect if the objective is on tacking zone
                beta = self.angle_saturation(ang = self.desired_heading - self.real_wind_angle, 
                                             min_ang=-180, 
                                             max_ang=180)
                
                to_tacking_zone_UW =  180-self.hyperparam[2]*0.5<alpha or 0.5*self.hyperparam[2]-180>alpha
                obj_is_tacking_UW =  180-self.hyperparam[2]*0.5<beta or 0.5*self.hyperparam[2]-180>beta 
                to_tacking_zone_TW =  alpha<self.hyperparam[4]/2 and alpha>-self.hyperparam[4]/2
                obj_is_tacking_TW = beta<self.hyperparam[4]/2 and beta>-self.hyperparam[4]/2
                
                if  to_tacking_zone_UW and obj_is_tacking_UW:                   
                    n3 = self.refresh_tack_direction(heading_angle = actual_heading, 
                                                     real_wind = self.real_wind_angle,
                                                     tack_angle=-self.tack_direction*self.hyperparam[5],
                                                     phase = 180)
                    
                    self.desired_heading = self.angle_saturation(ang = -self.tack_direction*self.hyperparam[5]+self.real_wind_angle+180, 
                                                                 min_ang=-180, 
                                                                 max_ang=180)

                    self.tack = True
                elif to_tacking_zone_TW and obj_is_tacking_TW:                   
                    n3 = self.refresh_tack_direction(heading_angle = actual_heading, 
                                                     real_wind = self.real_wind_angle,
                                                     tack_angle = self.tack_direction*self.hyperparam[13],
                                                     phase = 0)
                    self.desired_heading = self.angle_saturation(ang = self.tack_direction*self.hyperparam[13]+self.real_wind_angle, 
                                                                 min_ang=-180, 
                                                                 max_ang=180)
                    self.tack = True
                else:
                    self.tack_angle_logic [0] = -1000
                    self.tack_angle_logic [1] = -1000
                    self.tack = False
                    n3 = 0
                
                if self.desired_heading != 0 and self.waypoints[self.state+1][2] == 1:
                    self.desired_heading -= 180*self.desired_heading/np.abs(self.desired_heading)
                elif self.desired_heading == 0:
                    self.desired_heading = 180
                self.err_ang = (1-2*self.waypoints[self.state+1][2]) * self.angle_saturation(
                                                                   ang=self.desired_heading - actual_heading,
                                                                   min_ang=-180,
                                                                   max_ang=180)
                n3 = 0
                if abs(self.err_ang)>=self.sensor_max[0]:
                    self.err_ang = (self.sensor_max[0]-1)*self.err_ang/abs(self.err_ang)
                if abs(pitch)>=self.sensor_max[2]:
                    pitch = (self.sensor_max[2]-1)*pitch/abs(pitch)
                
                n1 = ((self.err_ang+self.sensor_max[0])*self.hyperparam[0])//(2*self.sensor_max[0])
                n2 = ((pitch+self.sensor_max[2])*self.hyperparam[1])//(2*self.sensor_max[2])
                l[int(self.hyperparam[0]*(self.hyperparam[1]*n3+n2)+n1)] = max_rate
                self.n_data.append(l)
                
            else:
                
                l = [min_rate]*int(2*self.hyperparam[7]);
                self.sum_angle = self.angle_saturation(ang = angle_apparent + actual_heading, #Is the apparent wind, no the real
                                                       min_ang=-180, 
                                                       max_ang=180)
                n = (self.hyperparam[7]*(self.sum_angle+180))//360
                n2 = self.waypoints[self.state+1][2]
                l[int(self.hyperparam[7]*n2+n)] = max_rate
                self.n_data.append(l)
                
    def control_inputs_Viel(self, data):
        self.distance = np.sqrt((data[1]-self.waypoints[self.state+1][0])**2+(data[2]-self.waypoints[self.state+1][1])**2)
        
        actual_heading = data[5] 
        self.real_wind_angle = self.angle_saturation(ang = data[6] + actual_heading, 
                                                min_ang=-180, 
                                                max_ang=180)
        self.actual_speed = np.sqrt(data[7]**2+data[8]**2)
        angle_speed = actual_heading#np.arctan2(data[8], data[7])
        angle_apparent = self.aparent_wind(real_wind_angle = self.real_wind_angle, 
                  sailboat_speed = self.actual_speed,
                  yaw = actual_heading)
        angle_apparent = self.angle_saturation(ang = angle_apparent - angle_speed, 
                                               min_ang=-180, 
                                               max_ang=180)
        #pitch = data[4]
        alpha = -data[6] # A variable for detect if the sailboat is on tacking zone
        self.desired_heading = 180*np.arctan2(self.waypoints[self.state+1][1]-data[2],
                                              self.waypoints[self.state+1][0]-data[1])/np.pi
          # A variable for detect if the objective is on tacking zone
        beta = self.angle_saturation(ang = self.desired_heading - self.real_wind_angle, 
                                     min_ang=-180, 
                                     max_ang=180)
        
        to_tacking_zone_UW =  180-self.hyperparam[2]*0.5<alpha or 0.5*self.hyperparam[2]-180>alpha
        obj_is_tacking_UW =  180-self.hyperparam[2]*0.5<beta or 0.5*self.hyperparam[2]-180>beta 
        to_tacking_zone_TW =  alpha<self.hyperparam[4]/2 and alpha>-self.hyperparam[4]/2
        obj_is_tacking_TW = beta<self.hyperparam[4]/2 and beta>-self.hyperparam[4]/2
        
        if  to_tacking_zone_UW and obj_is_tacking_UW:                   
            self.refresh_tack_direction(heading_angle = actual_heading, 
                                        real_wind = self.real_wind_angle,
                                        tack_angle=-self.tack_direction*self.hyperparam[5],
                                        phase = 180)
            
            self.desired_heading = self.angle_saturation(ang = -self.tack_direction*self.hyperparam[5]+self.real_wind_angle+180, 
                                                         min_ang=-180, 
                                                         max_ang=180)

            self.tack = True
        elif to_tacking_zone_TW and obj_is_tacking_TW:                   
            self.refresh_tack_direction(heading_angle = actual_heading, 
                                        real_wind = self.real_wind_angle,
                                        tack_angle = self.tack_direction*self.hyperparam[13],
                                        phase = 0)
            self.desired_heading = self.angle_saturation(ang = self.tack_direction*self.hyperparam[13]+self.real_wind_angle, 
                                                         min_ang=-180, 
                                                         max_ang=180)
            self.tack = True
        else:
            self.tack_angle_logic [0] = -1000
            self.tack_angle_logic [1] = -1000
            self.tack = False
  
       #delta_rmax,delta_smax,d_theta,theta,phi,sigma_awind,Vi_asterisc,epsilon
        return [np.radians(self.desired_heading),np.radians(actual_heading),angle_speed, np.radians(angle_apparent)]
    
    
    
    def control_inputs_PI(self, data):
            
        obj1 = self.waypoints[self.state+1][0]
        obj2 = self.waypoints[self.state+1][1]
        self.distance = np.sqrt((data[1]-self.waypoints[self.state+1][0])**2+(data[2]-self.waypoints[self.state+1][1])**2)
        
        actual_heading = data[5] 
        self.real_wind_angle = self.angle_saturation(ang = data[6] + actual_heading, 
                                                min_ang=-180, 
                                                max_ang=180)
        
        self.actual_speed = np.sqrt(data[7]**2+data[8]**2)
        angle_speed = actual_heading#np.arctan2(data[8], data[7])
        angle_apparent = self.aparent_wind(real_wind_angle = self.real_wind_angle, 
                  sailboat_speed = self.actual_speed,
                  yaw = actual_heading)
        angle_apparent = self.angle_saturation(ang = angle_apparent - angle_speed, 
                                               min_ang=-180, 
                                               max_ang=180)
        #pitch = data[4]
        alpha = -data[6] # A variable for detect if the sailboat is on tacking zone
        self.desired_heading = 180*np.arctan2(self.waypoints[self.state+1][1]-data[2],
                                              self.waypoints[self.state+1][0]-data[1])/np.pi
          # A variable for detect if the objective is on tacking zone
        beta = self.angle_saturation(ang = self.desired_heading - self.real_wind_angle, 
                                     min_ang=-180, 
                                     max_ang=180)
        
        to_tacking_zone_UW =  180-self.hyperparam[2]*0.5<alpha or 0.5*self.hyperparam[2]-180>alpha
        obj_is_tacking_UW =  180-self.hyperparam[2]*0.5<beta or 0.5*self.hyperparam[2]-180>beta 
        to_tacking_zone_TW =  alpha<self.hyperparam[4]/2 and alpha>-self.hyperparam[4]/2
        obj_is_tacking_TW = beta<self.hyperparam[4]/2 and beta>-self.hyperparam[4]/2
        
        if  to_tacking_zone_UW and obj_is_tacking_UW:                   
            self.refresh_tack_direction(heading_angle = actual_heading, 
                                        real_wind = self.real_wind_angle,
                                        tack_angle=-self.tack_direction*self.hyperparam[5],
                                        phase = 180)
            
            self.desired_heading = self.angle_saturation(ang = -self.tack_direction*self.hyperparam[5]+self.real_wind_angle+180, 
                                                         min_ang=-180, 
                                                         max_ang=180)

            self.tack = True
            obj1 = 100*np.cos(np.radians(self.desired_heading))+self.waypoints[self.state][0]
            obj2 = 100*np.sin(np.radians(self.desired_heading))+self.waypoints[self.state][1]
        elif to_tacking_zone_TW and obj_is_tacking_TW:                   
            self.refresh_tack_direction(heading_angle = actual_heading, 
                                        real_wind = self.real_wind_angle,
                                        tack_angle = self.tack_direction*self.hyperparam[13],
                                        phase = 0)
            self.desired_heading = self.angle_saturation(ang = self.tack_direction*self.hyperparam[13]+self.real_wind_angle, 
                                                         min_ang=-180, 
                                                         max_ang=180)
            self.tack = True
            obj1 = 100*np.cos(np.radians(self.desired_heading))+self.waypoints[self.state][0]
            obj2 = 100*np.sin(np.radians(self.desired_heading))+self.waypoints[self.state][1]
        else:
            self.tack_angle_logic [0] = -1000
            self.tack_angle_logic [1] = -1000
            self.tack = False
            
        position =  [data[1],data[2],obj1,obj2]
        angles = [data[3],data[4],data[5]]
       #delta_rmax,delta_smax,d_theta,theta,phi,sigma_awind,Vi_asterisc,epsilon
        return [True]+[position]+[angles]+[data[6]]
        
