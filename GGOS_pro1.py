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


    def __init__(self, iter_isdc = 0):
        self.iter_isdc = iter_isdc
    
    def set_iter_isdc(self, iter_isdc):
        self.iter_isdc = iter_isdc
        
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
        temp_years = []
        years_mjd = []
        facts = []
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
                            temp_years.append(float(k_o[0]))
                            years_mjd.append(k_o[4])
                            facts.append(k_o[5::])
                    except:
                        continue
        self.facts_isdc = np.array(facts).astype(float)
        self.years_isdc = temp_years
        self.years_mjd_isdc =years_mjd
        
                    
    def getter(self, isdc_tc, year, iterator):
        """ isdc_tc >>> Datei vom TC oder von ISDC
            year >>> Index 1. Wert des Jahres
            iterator >>> year + iterator >>> POS """
    
        if isdc_tc == "tc":
            index_year = self.years_tc.index(year)
            pos = index_year + iterator
            if self.years_tc[pos] == year:
                corrent_line = self.facts_tc[pos][1::]
            else:                
                return 0      
        
        elif isdc_tc == "isdc":
            index_year = self.years_isdc.index(year)
            pos = index_year + self.iter_isdc
            if pos <= self.facts_isdc.shape[0]-1 and self.years_isdc[pos] == year  :
                if (iterator%3) == 0:
                    pos = index_year + self.iter_isdc               
                elif (iterator%3) == 1:
                    pos = index_year + self.iter_isdc
                elif (iterator%3) == 2:
                    pos = index_year + self.iter_isdc
                    self.set_iter_isdc(self.iter_isdc + 1)
            else:
                return 0
            corrent_line = self.facts_isdc[pos,:]

        return corrent_line
        

                
    
#sun.txt    moon.txt     earthRotationVector.txt      
#potentialCoefficientsAOHIS.txt     potentialCoefficientsTides.txt

test = data()
test.read_given_txt("./lab01_data/",  "potentialCoefficientsTides.txt")
test.read_isdc_files("OAM", ['2005'])  
    
p = np.zeros([9000, 5])
g = np.zeros([9000, 6])
for i in range(9000):
    p[i,:] = test.getter("tc", 2005, i)
    g[i,:] = test.getter("isdc", 2005, i)
    





















































































































































   