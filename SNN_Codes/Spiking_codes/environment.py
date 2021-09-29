#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:14:17 2021

@author: nelson
"""
import numpy as np

waypoints = [
    [(250.0, 95.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(255.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(260.0, 105.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(265.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(270.0, 95.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(275.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(270.0, 105.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(265.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(260.0, 95.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(255.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(250.0, 105.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(245.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    [(240.0, 100.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
]

m=0
theta=0
b1=0
b2=0

def reward(data,tipo,limit):

    if tipo==1:
        l=[]
        v1=[]
        v2=[]
        l.append(data[0]-data[1])
        v1.append(limit)
        v2.append(0.2)
        r = normalize(data=l, vmax=v1, vmin=v2, A=1 ,B=-1)
    elif tipo==2:
        l=[]
        o=waypoints[len(waypoints)-1][0]
        v1=[]
        v2=[]
        l.append(np.sqrt((data[1]-data[3])**2+(data[2]-data[4])**2))
        v2.append(0)
        v1.append(2*np.sqrt((o[0]-data[3])**2+(o[1]-data[4])**2))
        r = normalize(data=l, vmax=v1, vmin=v2, A=-1 ,B=1)
    return r[0]


def normalize(data,vmax,vmin, A=1, B=0):
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

def carril_velero(ro,r,w):
    
    global m,theta,b1,b2
    
    m = (r[1]-ro[1])/(r[0]-ro[0])
    theta = np.arctan2(r[1]-ro[1], r[0]-ro[0])
    if(np.abs(theta)>np.pi/2):
        theta+=-np.pi*theta/np.abs(theta)
    b1 = r[1]-m*r[0]+0.5*w*(np.cos(theta)+m*np.sin(theta))
    b2 = r[1]-m*r[0]-0.5*w*(np.cos(theta)+m*np.sin(theta))
    
def is_restart(r,control_action):
    restart=2
    if r[1]<=m*r[0]+b1 and r[1]>=m*r[0]+b2:
        restart=control_action
    return restart

def get_plane(data):
    l=[]
    orig=[]
    orig.append(data[1])
    orig.append(data[2])
    l.append(data[3])
    l.append(data[4])

    return orig,l

def control_inputs(data,vmax,vmin, state, A, B):
    l=[]
    data[3] = waypoints[state][0][0]
    data[4] = waypoints[state][0][1]
    err_dist=np.sqrt((data[1]-data[3])**2+(data[2]-data[4])**2)
    err_ang=np.arctan2(data[4]-data[2],data[3]-data[1])-data[7]
    l.append(err_ang)
    l.append(data[5])
    l.append(data[6])
    l.append(data[8])
    
    l=normalize(l, vmax, vmin, A, B)
    return l,err_dist


