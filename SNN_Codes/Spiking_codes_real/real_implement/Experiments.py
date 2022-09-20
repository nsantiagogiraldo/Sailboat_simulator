#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:54:21 2022

@author: nelson
"""

import SNN_train_controller as ctrl

m = ctrl.SNN_complete_train_test()
#m.test_PI()
# m.test_Viel2019()
# fail = False
# m = ctrl.SNN_complete_train_test()
# fail = m.train_SNN()
# del m
# m = ctrl.SNN_complete_train_test()
# if not fail:
#m.test_SNN()
m.test_SNN_real()
