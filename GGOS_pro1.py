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
    G = 6.674*10**(-11)
    k_Re = 0.3077
    k_Im = 0.0036
    GM_sun = 1.32712442076*10**(20)
    GM_moon = 4.9027779*10**(12)
    M_earth = 5.9737*10**(24)
    R_earth = 6378136.6
    A_B_strich = 0.3296108 * M_earth * R_earth**2
    C_strich = 0.3307007 * M_earth * R_earth**2
    Omega_n = 7.2921151467064*10**(-5)


    def __init__(self):
        pass
    
    def read_given_txt(self, path, filename):
        facts = []
        years = []
        years_mjd = []
        with open(path + filename) as f:
            file = f.read().splitlines()
            for k in range(2,len(file)-1):
                temp_file = file[k].split()
                temp_years = ju.from_jd(float(temp_file[0]), fmt = 'mjd')
                years.append(temp_years.year)
                years_mjd.append(temp_file[0])
                facts.append(temp_file)
            self.facts_tc = np.asarray(facts).astype(float)
            self.years_tc = years
            self.years_mjd_tc = years_mjd
    
    def read_isdc_files(self, path, years = ['2005', '2006', '2007', '2008',
                                                 '2009', '2010', '2011', '2012',
                                                 '2013', '2014', '2015'] ):
        line = []
        for y in years:
            if path == "AAM":
                path_folder = "./AAM/"
                file_name = "ESMGFZ_AAM_v1.0_03h_" + str(y) + ".asc"
            elif path == "HAM":
                path_folder = "./HAM/"
                file_name = "ESMGFZ_HAM_v1.2_24h_" + str(y) + ".asc"
            elif path == "OAM":
                path_folder = "./OAM/"
                file_name = "ESMGFZ_OAM_v1.0_03h_" + str(y) + ".asc"
            elif path == "SLAM":
                path_folder = "./SLAM/"
                file_name = "ESMGFZ_SLAM_v1.2_24h_" + str(y) + ".asc"
                

            with open(path_folder + file_name, 'r') as f:
                file = f.read().splitlines()
                for k in range(len(file)):
                    k_o = file[k].split()
                    try:                                # weil str !>> int
                        if k_o[0] in years:
                            line.append(k_o)
                    except:
                        continue
        self.facts_isdc = np.array(line).astype(float)
                    
    def getter(self, isdc_tc, year, iterator):
        """ isdc_tc >>> Datei vom TC oder von ISDC
            year >>> Index 1. Wert des Jahres
            iterator >>> year + iterator >>> POS """
        if isdc_tc == "tc":
            index_year = self.years_tc.index(year)
            pos = index_year + iterator
            
        elif isdc_tc == "isdc":
            pass
        
        

                
    
#sun.txt    moon.txt     earthRotationVector.txt      
#potentialCoefficientsAOHIS.txt     potentialCoefficientsTides.txt

test = data()
test.read_given_txt("./lab01_data/",  "potentialCoefficientsTides.txt")
test.read_isdc_files("OAM", ['2005', '2006', '2010'])  
    
test.getter("tc", 2006, 0)





















































































































































   