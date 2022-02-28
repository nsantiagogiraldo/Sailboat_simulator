#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:54:21 2022

@author: nelson
"""

import SNN_train_controller as ctrl

fail = False
m = ctrl.SNN_complete_train_test()
fail = m.train_SNN()
m = ctrl.SNN_complete_train_test()
if not fail:
    m.test_SNN()