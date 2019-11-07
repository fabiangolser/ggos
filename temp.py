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

class Data():


    def __init__(self,):
        self.read_isdc_files()

    
    
    def read_isdc_files(self, path = ["./AAM/", "./HAM/", "./OAM/", "./SLAM/"]):
        facts = []
        years = ['2005', '2006', '2007', '2008', '2009',
                 '2010', '2011', '2012', '2013', '2014', '2015']
        for p in path:
            for i in os.listdir(p):
                with open(p + i, 'r') as f:
                    file = f.read().splitlines()
                    header = 0
                    for k in range(0, len(file)):
                        k_o = file[k].split()
                        if len(k_o) != 0 and k_o[0] in years:
                            facts.append(k_o)
                        else:
                            header += 1
                            
            if p == "./AAM/":
                self.aam_ = np.asarray(facts).astype(float)
            if p == "./HAM/":
                self.ham_ = np.asarray(facts).astype(float)
            if p == "./OAM/":
                self.oam_ = np.asarray(facts).astype(float)
            if p == "./SLAM/":
                self.slam_ = np.asarray(facts).astype(float)
                
            facts.clear()
        
test = Data()



x = np.linspace(0,87649, 4017, endpoint = True)
y = test.ham_[:,5]

val = np.linspace(0,87649, 87649, endpoint = True, dtype = int)


yinterp = np.interp(val, x, y)

fig, ax = plt.subplots(1)
fig.suptitle('slam')
ax.set_title('Anzahl der verliehenen Fahrräder')
plt.plot(val, yinterp, linewidth = 0.25, color = 'hotpink', marker = '.')
plt.plot(x, y ,'b+', linewidth = 0.1)
        
        
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


    def __init__(self,):
        self.read_given_txt()
        self.read_isdc_files()
        self.interpolate()
        
    def read_given_txt(self):
        facts = []
        self.years_tc = []
        path = "./lab01_data/"
        for i in os.listdir(path):
            with open(path + i) as f:
                file = f.read().splitlines()
                for k in range(2,len(file)-1):
                    temp_file = file[k].split()
                    temp_years = ju.from_jd(float(temp_file[0]), fmt = 'mjd')
                    
                if i[:-4] == 'moon':
                    self.years_tc.append(temp_years.year)
                    self.moon = facts.append(temp_file)  
                if i[:-4] == "potentialCoefficientsAOHIS":
                    self.pc_aohis_ = np.asarray(facts).astype(float)
                if i[:-4] == 'potentialCoefficientsTides':
                    self.pc_tides_ = np.asarray(facts).astype(float)
                if i[:-4] == 'earthRotationVector':
                    self.earth_rottation_ = np.asarray(facts).astype(float)
                if i[:-4] == 'moon':
                    self.moon_ = np.asarray(facts).astype(float)
                if i[:-4] =='sun':
                    self.sun_ = np.asarray(facts).astype(float)
                
                facts.clear()

    
    
    def read_isdc_files(self, path = ["./AAM/", "./HAM/", "./OAM/", "./SLAM/"]):
        facts = []
        years = ['2005', '2006', '2007', '2008', '2009',
                 '2010', '2011', '2012', '2013', '2014', '2015']
        for p in path:
            for i in os.listdir(p):
                with open(p + i, 'r') as f:
                    file = f.read().splitlines()
                    header = 0
                    for k in range(0, len(file)):
                        k_o = file[k].split()
                        if len(k_o) != 0 and k_o[0] in years:
                            facts.append(k_o)
                        else:
                            header += 1
                            
            if p == "./AAM/":
                self.aam = np.asarray(facts).astype(float)
            if p == "./HAM/":
                self.ham = np.asarray(facts).astype(float)
            if p == "./OAM/":
                self.oam = np.asarray(facts).astype(float)
            if p == "./SLAM/":
                self.slam = np.asarray(facts).astype(float)
                
            facts.clear()
    def interpolate(self):
        isdc = [self.aam, self.oam, self.ham, self.slam]
        for i in isdc:
            size = i.shape
        
    def getter_tc(self, year, indizes, file):
        index_year = self.years_tc.index(year)
        pos = index_year + indizes
        corrent_line = file[pos][1::]
        return corrent_line      
        
    def getter_isdc_3H(self, year, indizes, file):
        years_isdc = file[:,0].tolist()
        index_year = years_isdc.index(year)
        pos = index_year + indizes   
        corrent_line = file[pos,:]           
#        corrent_line = file[pos,5::]
        return corrent_line

    def moon(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.moon_)

    def sun(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.sun_)
                
    def earth_rottation(self,hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.earth_rottation_)

    def pc_aohis(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.pc_aohis_)

    def pc_tide(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.pc_tides_)
    
    def aam(self, hour_since_2005):
        offset = 2920
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), int(np.floor(hour_since_2005/3)) % offset, self.aam_)

    def aom(self, hour_since_2005):
        offset = 2920
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), int(np.floor(hour_since_2005/3))% offset, self.aom_)
    
    def ham(self, hour_since_2005):
        offset = 365
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), int(np.floor(hour_since_2005/24))% offset, self.ham_)

    def slam(self, hour_since_2005):
        offset = 365
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), int(np.floor(hour_since_2005/24))% offset, self.slam_)

#sun.txt    moon.txt     earthRotationVector.txt      
#potentialCoefficientsAOHIS.txt     potentialCoefficientsTides.txt

test = data()
#z = 10000
#s = 12
#a = np.zeros([z, 3])
#b = np.zeros([z, 3])
#c = np.zeros([z, 3])
#d = np.zeros([z, 5])
#e = np.zeros([z, 5])
#f = np.zeros([z, 6])
#g = np.zeros([z, s])
#h = np.zeros([z, s])
#j = np.zeros([z, 3])
#for i in range(10000):
#    a[i,:] = np.transpose(test.moon(i))
#    b[i,:] = np.transpose(test.sun(i))
#    c[i,:] = np.transpose(test.earth_rottation(i))
#    d[i,:] = np.transpose(test.pc_aohis(i))
#    e[i,:] = np.transpose(test.pc_tide(i))
#    f[i,:] = np.transpose(test.aam(i))
##    g[i,:] = test.aom(i)
##    h[i,:] = test.ham(i)
##    j[i,:] = np.transpose(test.slam(i))
  





















































































































































           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        