#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:43:54 2022

@author: nelson
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(200,280,0.1)

while True:
    
    m= float(input("m:"))
    b1 = float(input('b1:'))
    b2 = float(input('b2:'))
    
    y1 = m*x + b1
    y2 = m*x +b2
    
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Navigation channels')
    plt.show()
