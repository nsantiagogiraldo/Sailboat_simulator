#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:35:30 2022

@author: nelson
"""
path = '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/config'
out_rudder=[5,9,11,13]
out_sail=[11,13,15,17]
max_error_rudder=[70,60,50,40]
max_error_sail=[70,60,50,40]
input_rudder=[5,10]
input_sail=[36,18]
name = 'config'
i=0

for a in out_rudder:
    for b in out_sail:
        for c in max_error_rudder:
            for d in max_error_sail:
                for e in input_rudder:
                    for f in input_sail:
                        i+=1
                        rudder_init = 2*c//e
                        file = open(path+'/'+name+str(i)+'.txt','w')
                        text = (str(c)+', '+str(d)+', '+'30\n'+str(-c)+', '+str(-d)+', '+'-30\n'
                                +'45, 90\n-45, -90\n'
                                +str(rudder_init)+', 1, 120, '+str(a)+', 60, 70, 4, '+str(f)+', '+str(b)+', 14, 100, 6, 0.35, 45, 10, 10, 3\n'
                                +'rudder,sail\n2, 2\n0, 0\n1\n0\n240\npoisson\n10\n0\n1\n500\n1\n1\n0\n'+str(i)
                                )
                        file.write(text)
                    
                        file.close()
