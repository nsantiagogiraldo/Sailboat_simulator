#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:01:25 2022

@author: nelson
"""

name = '/home/nsantiago/trained.txt'
file = open(name,'r')
info = file.read().split('\n')
file.close()
number_test = []
simulate = ''

for i in info:
    if i.find('Test') == 0:
        u_file = i[5:-4]
        number_test.append(int(u_file))

for i in range(int(info[0])):
    if number_test.count(i+1) == 0:
        simulate+=(str(i+1)+'\n')

file = open(name,'w')
file.write(simulate)
file.close()
