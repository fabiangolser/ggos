# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:46:19 2019

@author: Felix
"""

import sys
import numpy as np
import GGOS_pro1 as g_data

class RotationModell():
    
    omega_0 = 123
    
    def __init__(self, data):
        pass
    
    def m_of_t(self, j):
        """Fromel!!!!!!!!!! """
        return np.zeros([3, ])
    
    def force(self, j):
        """Fromel!!!!!!!!!! """
        pass
    
    def tg_of_t(self):
        pass
    
    def tr_of_t(self):
        pass
    
    def t_total(self, tg, tr):
        pass
    
    def h_it_self(self):
        pass
    
    def delta_t(self):
        pass
    
    def delta_h(self):
        pass
    
    def omega_dot(self, m = np.zeros([3, ]), dt = np.zeros([3, 3]), h = np.zeros([3, ]), dh = np.zeros([3, ])):
        pass