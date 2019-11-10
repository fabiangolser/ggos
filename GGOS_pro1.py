"""
GGOS Lab

This is the class used encapsulates the data.
All data is read from different files and stored inside this class.
"""
import numpy as np
import julian as ju
import os


class Data:
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
        self.w = np.array([self.earth_rotation(0), ])
        self.w_dot = np.array([[0, 0, 0], ])    # maybe not initialized correct yet

    def append_w(self, w_new):
        """appends the current w(x, y, z)"""
        self.w = np.append(self.w, w_new, axis=0)

    def append_w_dot(self, w_dot_new):
        """appends the current w_dot(x, y, z)"""
        self.w_dot = np.append(self.w_dot, w_dot_new, axis=0)

    def read_given_txt(self):
        facts = []
        self.years_tc = []
        path = "./lab01_data/"
        for i in os.listdir(path):
            with open(path + i) as f:
                file = f.read().splitlines()
                for k in range(2,len(file)-1):
                    temp_file = file[k].split()
                    facts.append(temp_file)
                    if i == 'moon.txt':
                        temp_years = ju.from_jd(float(temp_file[0]), fmt = 'mjd')
                        self.years_tc.append(temp_years.year)
                    
                if i[:-4] == "potentialCoefficientsAOHIS":
                    self.pc_aohis_ = np.asarray(facts).astype(float)
                if i[:-4] == 'potentialCoefficientsTides':
                    self.pc_tides_ = np.asarray(facts).astype(float)
                if i[:-4] == 'earthRotationVector':
                    self.earth_rotation_ = np.asarray(facts).astype(float)
                if i[:-4] == 'moon':
                    self.moon_ = np.asarray(facts).astype(float)
                if i[:-4] =='sun':
                    self.sun_ = np.asarray(facts).astype(float)
                facts.clear()

    def read_isdc_files(self, path = ["./AAM/", "./HAM/", "./OAM/", "./SLAM/"]):
        facts = []
        years = ['2005', '2006', '2007', '2008', '2009',
                 '2010', '2011', '2012', '2013', '2014']
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
                self.aam_f = np.asarray(facts).astype(float)
            if p == "./HAM/":
                self.ham_f = np.asarray(facts).astype(float)
            if p == "./OAM/":
                self.oam_f = np.asarray(facts).astype(float)
            if p == "./SLAM/":
                self.slam_f = np.asarray(facts).astype(float)
            facts.clear()
            
    def interpolate(self):
        self.aam_ = np.zeros([87649, 6])
        self.oam_ = np.zeros([87649, 6])
        self.ham_ = np.zeros([87649, 6])
        self.slam_ = np.zeros([87649, 3])
        
        isdc = [self.aam_f, self.oam_f, self.ham_f, self.slam_f]
        for i in isdc:
            copy = i[:,5::]
            
            size = copy.shape
            x = np.linspace(0, 87649, size[0], endpoint = True, dtype = int)
            for j in range(size[1]):
                y = copy[:,j]
                val = np.linspace(0,87649, 87649, endpoint = True, dtype = int)
                yinterp = np.interp(val, x, y)
                if i[0,-1] == self.aam_f[0,-1]:
                    self.aam_[:,j] = yinterp
                if i[0,-1] == self.oam_f[0,-1]:
                    self.oam_[:,j] = yinterp
                if i[0,-1] == self.ham_f[0,-1]:
                    self.ham_[:,j] = yinterp
                if i[0,-1] == self.slam_f[0,-1]:
                    self.slam_[:,j] = yinterp

    def getter_tc(self, year, indizes, file):
        index_year = self.years_tc.index(year)
        pos = index_year + indizes
        corrent_line = file[pos][1::]
        return corrent_line      
        
    def getter_isdc_3H(self, year, indizes, file):
        index_year = self.years_tc.index(year)
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
                
    def earth_rotation(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.earth_rotation_)

    def pc_aohis(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.pc_aohis_)

    def pc_tide(self, hour_since_2005):
        offset = 8760
        return self.getter_tc(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.pc_tides_)
    
    def aam(self, hour_since_2005):
        offset = 8760
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.aam_)

    def aom(self, hour_since_2005):
        offset = 8760
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.oam_)
    
    def ham(self, hour_since_2005):
        offset = 8760
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.ham_)

    def slam(self, hour_since_2005):
        offset = 8760
        return self.getter_isdc_3H(2005+int(np.floor(hour_since_2005/offset)), hour_since_2005 % offset, self.slam_)
