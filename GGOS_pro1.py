# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
import julian as ju
import matplotlib.pyplot as plt
import sys
import os
import math as m
print(sys.version)

class data():
    
    def __init__(self):
        pass
    
    def read_given_txt(self, path, filename):
        line = []
        with open(path + filename) as f:
            file = f.read().splitlines()
            for k in range(2,len(file)-2):
                line.append(file[k].split())
        return np.asarray(line).astype(float)
    
    def read_isdc_files(self, path, year):
        line = []
        if path == "AAM":
            path_folder = "./AAM/"
            file_name = "ESMGFZ_AAM_v1.0_03h_" + str(year) + ".asc"
        elif path == "HAM":
            path_folder = "./HAM/"
            file_name = "ESMGFZ_HAM_v1.2_24h_" + str(year) + ".asc"
        elif path == "OAM":
            path_folder = "./OAM/"
            file_name = "ESMGFZ_OAM_v1.0_03h_" + str(year) + ".asc"
        elif path == "SLAM":
            path_folder = "./SLAM/"
            file_name = "ESMGFZ_SLAM_v1.2_24h_" + str(year) + ".asc"

        with open(path_folder + file_name, 'r') as f:
            file = f.read().splitlines()
            for k in range(len(file)):
                k_o = file[k].split()
                try:                                # weil str !>> int
                    if float(k_o[0]) == year:
                        line.append(file[k].split())
                except:
                    continue
        return np.array(line).astype(float)
    
    def read_isdc(self, path):
        line = []
        files = []
        
        for i in os.listdir(path):
            files.append(i)
            with open(path + i, 'r') as f:
                file = f.read().splitlines()
                for k in range(len(file)):
                    k_o = file[k].split()
                    try:
                        if k_o[0] in ['2005', '2006', '2007', '2008', '2009', '2010'
                                      '2011', '2012', '2013', '2014', '2015']:
                            line.append(file[k].split())
                    except:
                        continue
        return np.array(line).astype(float)
                    

                
                


#sun.txt    moon.txt     earthRotationVector.txt      
#potentialCoefficientsAOHIS.txt     potentialCoefficientsTides.txt
daten = data()
sun = daten.read_given_txt("./lab01_data/", "earthRotationVector.txt") 

effective_angular_momentum = daten.read_isdc_files("SLAM", 2005)

eimfach_alles = daten.read_isdc("./SLAM/")


#for i in range(len(sun)):
#    print(ju.from_jd(sun[i][0], fmt = 'mjd'))












































































































































   