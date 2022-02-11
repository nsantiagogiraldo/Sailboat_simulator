#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:35:28 2022

@author: nelson
"""

class text_files:
    
    name = ''
    columns_names = ''
    num_data = 0
    path = ''

    def __init__(self, new_file, file_name, path ,structure = ''):
        self.path = path+'/data_result/'
        self.name = self.path+file_name+'.csv'
        if new_file:
            self.columns_names = structure
            self.new_file()
        else:
            self.load_file()
            
    def new_file(self):
        file = open(self.name,'w')
        header = self.convert_data(self.columns_names).replace('0', '')
        file.write(header)
        file.close()
        
    def load_file(self):
        file = open(self.name,'r')
        data = file.read().split('\n')
        self.columns_names = data[0].split(',')
        self.columns_names.pop(0)
        self.columns_names.pop(-1)
        value = data[-2].split(',')[0]
        if value == '':
            value = 0
        self.num_data = int(value)+1
        file.close()
        
    def convert_data(self,data):
        str_data = str(self.num_data)+','
        for i in data:
            str_data += str(i)+','
        str_data += '\n'
        self.num_data += 1
        return str_data
    
    def append_data(self,data):
        file = open(self.name,'a')  
        str_data = self.convert_data(data)
        file.write(str_data)
        file.close()

    def is_created(self):
        filePath = self.name
        try:
            with open(filePath, 'r') as f:
                return True
        except:
            return False 
