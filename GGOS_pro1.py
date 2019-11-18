"""
GGOS Lab

This is the class used encapsulates the data.
All data is read from different files and stored inside this class.
"""
import numpy as np
import julian as ju
import os

from xdg.Menu import Layout

import ggosPlot as g_plot
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

special_computation = True
#special_computation = False

plt.close('all')

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

    def __init__(self):
        self.read_given_txt()
        self.read_isdc_files()
        self.interpolate()
        self.w = np.array([self.earth_rotation(0), ])
        self.w_dot = np.array([[0, 0, 0], ])

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

    def read_isdc_files(self, path = ["./AAM/", "./HAM/",
                                      "./OAM/", "./SLAM/"]):
        facts = []
        years = ['2005', '2006', '2007', '2008', '2009',
                 '2010', '2011', '2012', '2013', '2014']
#        for k in os.listdir("./AAM")
        for p in path:
            for i in sorted(os.listdir(p)):
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
            copy = i[:, 5::]
            
            size = copy.shape
            x = np.linspace(0, 87649, size[0], endpoint = True, dtype = int)
            for j in range(size[1]):
                y = copy[:,j]
                val = np.linspace(0,87649, 87649, endpoint = True, dtype = int)
                yinterp = np.interp(val, x, y)
                if i[0,-1] == self.aam_f[0,-1]:
                    self.aam_[:, j] = yinterp
                if i[0,-1] == self.oam_f[0,-1]:
                    self.oam_[:, j] = yinterp
                if i[0,-1] == self.ham_f[0,-1]:
                    self.ham_[:, j] = yinterp
                if i[0,-1] == self.slam_f[0,-1]:
                    self.slam_[:, j] = yinterp

    def getter_tc(self, indizes, file):
        pos = indizes
        corrent_line = file[pos][1::]
        return corrent_line      
        
    def getter_isdc_3H(self, indizes, file):
        pos = indizes
        corrent_line = file[pos,:]
        return corrent_line

    def moon(self, hour_since_2005):
        return self.getter_tc(hour_since_2005, self.moon_)

    def sun(self, hour_since_2005):
        return self.getter_tc(hour_since_2005, self.sun_)
                
    def earth_rotation(self, hour_since_2005):
        return self.getter_tc(hour_since_2005, self.earth_rotation_)

    def pc_aohis(self, hour_since_2005):
        return self.getter_tc(hour_since_2005, self.pc_aohis_)

    def pc_tide(self, hour_since_2005):
        return self.getter_tc(hour_since_2005, self.pc_tides_)
    
    def aam(self, hour_since_2005):
        return self.getter_isdc_3H(hour_since_2005, self.aam_)

    def aom(self, hour_since_2005):
        return self.getter_isdc_3H(hour_since_2005, self.oam_)
    
    def ham(self, hour_since_2005):
        return self.getter_isdc_3H(hour_since_2005, self.ham_)

    def slam(self, hour_since_2005):
        return self.getter_isdc_3H(hour_since_2005, self.slam_)
#______________________________________________________________________________
test = Data()
z = len(test.moon_)
G = 6.674*10**(-11)
GM_sun = 1.32712442076*10**(20)
GM_moon = 4.9027779*10**(12)
if special_computation:
    G = 6.674 * 10 ** (-11) / 2
    GM_sun = 1.32712442076 * 10 ** (20) / 2
    GM_moon = 4.9027779 * 10 ** (12) / 2
k_Re = 0.3077
k_Im = 0.0036
M_earth = 5.9737*10**(24)
R_earth = 6378136.6
A_B_strich = 0.3296108 * M_earth * R_earth**2
C_strich = 0.3307007 * M_earth * R_earth**2
Omega_n = 7.2921151467064*10**(-5)

moon = np.zeros([z, 3])
sun = np.zeros([z, 3])
earth_rot = np.zeros([z, 3])
pc_aohis = np.zeros([z, 5])
pc_tide = np.zeros([z, 5])
aam = np.zeros([z, 6])
oam = np.zeros([z, 6])
ham = np.zeros([z, 6])
j = np.zeros([z, 3])

for i in range(z):
    moon[i, :] = np.transpose(test.moon(i))
    sun[i, :] = np.transpose(test.sun(i))
    earth_rot[i, :] = np.transpose(test.earth_rotation(i))
    pc_aohis[i, :] = np.transpose(test.pc_aohis(i))
    pc_tide[i, :] = np.transpose(test.pc_tide(i))
    aam[i, :] = np.transpose(test.aam(i))
    oam[i, :] = np.transpose(test.aom(i))
    ham[i, :] = np.transpose(test.ham(i))
  
aam[:, 3] = aam[:, 3] + -9.203219554984803*10**(-10) # e-10
aam[:, 4] = aam[:, 4] + -1.496376584857922*10**(-8) #e-08
aam[:, 5] = aam[:, 5] + 2.734436408977633*10**(-8) #e-08

oam[:, 3] = oam[:, 3] + -3.220254498203050*10**(-8) #e-08
oam[:, 4] = oam[:, 4] + 8.226002041060833*10**(-8) #e-08
oam[:, 5] = oam[:, 5] + 3.462197020095727*10**(-9) #e-09


########################################################################################################################
if special_computation:
    w = [0, 0, 0]
    w_v = np.zeros([z, 3])
    w_v[0, :] = earth_rot[0, :]
    w_dot_v = np.zeros([z, 3])
    tg_v = np.zeros([z, 3, 3])
    tr_v = np.zeros([z, 3, 3])
    h_v = np.zeros([z, 3])
    delta_t = np.zeros([3, 3])
    delta_h = np.zeros([3, ])
    polar_v = np.zeros([z, 2])
    delta_lod_v = np.zeros([z, ])
    polar_v_ref = np.zeros([z, 2])
    delta_lod_v_ref = np.zeros([z, ])

    # ______________________________________________________________________________
    # main loop
    for index in range(z - 1):
        # ___ sun
        M_sun = np.zeros([3, ])
        body = sun[index]
        GM = GM_sun
        r = np.sqrt(body[0] ** 2 + body[1] ** 2 + body[2] ** 2)
        scalar = 3 * GM / r ** 5
        M_sun[0] = body[1] * body[2] * (C_strich - A_B_strich)
        M_sun[1] = body[0] * body[2] * (A_B_strich - C_strich)
        M_sun[2] = body[0] * body[1] * (A_B_strich - A_B_strich)
        M_sun = scalar * M_sun

        # ___ moon
        M_moon = np.zeros([3, ])
        body = moon[index]
        GM = GM_moon
        r = np.sqrt(body[0] ** 2 + body[1] ** 2 + body[2] ** 2)
        scalar = 3 * GM / r ** 5
        M_moon[0] = body[1] * body[2] * (C_strich - A_B_strich)
        M_moon[1] = body[0] * body[2] * (A_B_strich - C_strich)
        M_moon[2] = body[0] * body[1] * (A_B_strich - A_B_strich)
        M_moon = scalar * M_moon

        # ___ sum
        m = np.zeros([3, ])
        m = M_moon + M_sun
        # ______________________________________________________________________________
        """ T_G(t) = sqrt(5/3)*M*R^2*(matrix) """
        scalar = np.sqrt(5 / 3) * M_earth * R_earth ** (2)
        # print("scalar = {}".format(scalar))

        matrix = np.zeros([3, 3])
        c_s = pc_aohis[index]
        c_s = c_s + pc_tide[index]
        # print("c_s = {}".format(c_s))
        matrix[0, 0] = (np.sqrt(1 / 3) * c_s[0]) - c_s[3]
        matrix[0, 1] = -c_s[4]
        matrix[0, 2] = -c_s[1]
        matrix[1, 0] = -c_s[4]
        matrix[1, 1] = (np.sqrt(1 / 3) * c_s[0]) + c_s[3]
        matrix[1, 2] = -c_s[2]
        matrix[2, 0] = -c_s[1]
        matrix[2, 1] = -c_s[2]
        matrix[2, 2] = -(2 * np.sqrt(1 / 3) * c_s[0])
        # print("matrix = \n{}".format(matrix))

        # tr_
        tr_ = A_B_strich + A_B_strich + C_strich
        # print("tr_ = {}".format(tr_))
        matrix_2 = np.eye(3)
        matrix_2 = matrix_2 * (tr_ / 3)
        # print("matrix_2 = \n{}".format(matrix_2))

        tg = (scalar * matrix) + matrix_2
        tg_v[index, :] = tg

        # ______________________________________________________________________________
        """ T_R(t) = (O_N*R^5)/(3*G) * (matrix) """
        tr_scalar = (Omega_n * R_earth ** (5)) / (3 * G)
        # print("scalar = {}".format(scalar))

        # matrix
        tr_matrix = np.zeros([3, 3])
        w_x = w_v[index][0]
        w_y = w_v[index][1]

        tr_matrix[0, 2] = (k_Re * w_x + k_Im * w_y)
        tr_matrix[1, 2] = (k_Re * w_y - k_Im * w_x)
        tr_matrix[2, 0] = (k_Re * w_x + k_Im * w_y)
        tr_matrix[2, 1] = (k_Re * w_y - k_Im * w_x)
        # print("matrix = \n{}".format(matrix))

        tr = tr_scalar * tr_matrix
        tr_v[index, :] = tr

        # ______________________________________________________________________________
        """ T(t) = T_G(t) + T_R(t) """
        t = tg_v[index, :] + tr_v[index, :]

        # ______________________________________________________________________________
        # h
        h_x = aam[index, :] + oam[index, :] + ham[index, :]

        h = np.zeros([3, ])
        # h[0] = ((Omega_n * (C_strich - A_B_strich)) / 1.610) * h_x[3]
        # h[1] = ((Omega_n * (C_strich - A_B_strich)) / 1.610) * h_x[4]
        # h[2] = ((Omega_n *  C_strich) / 1.125) * h_x[5]

        h[0] = (1.610 / (Omega_n * (C_strich - A_B_strich))) * h_x[3]
        h[1] = (1.610 / (Omega_n * (C_strich - A_B_strich))) * h_x[4]
        h[2] = (1.125 / (Omega_n * C_strich)) * h_x[5]

        h_v[index, :] = h / 3600

        # ______________________________________________________________________________
        """ dTG(t) = T_G(t) - T_R(t-1) """
        if index > 0:
            delta_t = tg_v[index] - tg_v[index - 1]
        # else:
        #    delta_t = tg_v[index] - 0
        delta_t = delta_t / 3600

        # ______________________________________________________________________________
        """ dTG(t) = T_G(t) - T_R(t-1) """
        if index > 0:
            delta_h = h_v[index] - h_v[index - 1]
        else:
            delta_h = h_v[index] / 2
        delta_h = delta_h / 3600

        # ______________________________________________________________________________
        """ force """
        f_scalar = (Omega_n ** 2 * R_earth ** (5)) / (3 * G)

        # matrix
        f_matrix = np.zeros([3, 3])
        f_matrix[0, 0] = k_Re
        f_matrix[0, 1] = k_Im
        f_matrix[1, 0] = -k_Im
        f_matrix[1, 1] = k_Re
        # print("matrix = \n{}".format(matrix))

        t_temp = f_scalar * f_matrix
        f = t + t_temp
        # ______________________________________________________________________________
        # w_dot

        w = w_v[index, :]
        """ F^(-1) """
        f_invers = np.linalg.inv(f)

        """ (D*T_G/Dt) * w """
        dT_G_Dt_w = np.dot(delta_t, w)
        # print('delta_t({}) = \n{}'.format(index, delta_t))

        """ w x (T * w) """
        tw = np.dot(t, w)
        w_x_tw = np.cross(w, tw)

        """ w x h(t) """
        w_x_h = np.cross(w, h_v[index, :] - delta_h)

        """ w_dot = F^(-1) * [M - ((D*T_G/Dt) * w) - (w x (T * w)) - (w x h) - (D_h/Dt)] """
        """ w_dot = F^(-1) * [M - (DT_G_Dt_w)      - (w_x_Tw)      - (w x h) - (D_h/Dt)] """
        w_dot = np.dot(f_invers, (m - dT_G_Dt_w - w_x_tw - w_x_h))
        w_dot_v[index, :] = w_dot
        # print('w_dot({}) = \n{}'.format(index, w_dot_v[index, :]))

        w = w + w_dot * 3600
        w_v[index + 1, :] = w
        # print('w_v({}) = \n{}'.format(index, w_v[index, :]))

        # ______________________________________________________________________________
        # polar_motion
        """ x_p(t) = (R/W_N) * w_x(t), y_p(t) = (R/W_N) * w_y(t) """
        w = w_v[index]

        x_p = (R_earth / Omega_n) * w[0]
        y_p = (R_earth / Omega_n) * w[1]
        polar_v[index][0] = x_p
        polar_v[index][1] = y_p

        # polar_motion ref
        w = earth_rot[index]
        polar_v_ref[index][0] = (R_earth / Omega_n) * w[0]
        polar_v_ref[index][1] = (R_earth / Omega_n) * w[1]

        # ______________________________________________________________________________
        # delta_lod
        delta_lod_v[index] = 86400 * ((Omega_n - w_v[index][2]) / Omega_n)

        # delta_lod ref
        delta_lod_v_ref[index] = 86400 * ((Omega_n - earth_rot[index][2]) / Omega_n)

    # Don't touch it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ########################################################################################################################
    # save special computation matrices, restore original values and rerun simulation
    w_v_g_half = w_v
    G = 6.674 * 10 ** (-11)
    GM_sun = 1.32712442076 * 10 ** (20)
    GM_moon = 4.9027779 * 10 ** (12)


########################################################################################################################
w = [0, 0, 0]
w_v = np.zeros([z, 3])
w_v[0, :] = earth_rot[0, :]
w_dot_v = np.zeros([z, 3])
tg_v = np.zeros([z, 3, 3])
tr_v = np.zeros([z, 3, 3])
h_v = np.zeros([z, 3])
delta_t = np.zeros([3, 3])
delta_h = np.zeros([3, ])
polar_v = np.zeros([z, 2])
delta_lod_v = np.zeros([z, ])
polar_v_ref = np.zeros([z, 2])
delta_lod_v_ref = np.zeros([z, ])

#______________________________________________________________________________
# main loop
for index in range(z-1):
    #___ sun
    M_sun = np.zeros([3, ])
    body = sun[index]
    GM = GM_sun
    r = np.sqrt(body[0]**2 + body[1]**2 + body[2]**2)
    scalar = 3 * GM / r**5
    M_sun[0] = body[1] * body[2] * (C_strich   - A_B_strich)
    M_sun[1] = body[0] * body[2] * (A_B_strich - C_strich)
    M_sun[2] = body[0] * body[1] * (A_B_strich - A_B_strich)
    M_sun = scalar * M_sun
    
    #___ moon
    M_moon = np.zeros([3, ])
    body = moon[index]
    GM = GM_moon
    r = np.sqrt(body[0]**2 + body[1]**2 + body[2]**2)
    scalar = 3 * GM / r**5
    M_moon[0] = body[1] * body[2] * (C_strich -   A_B_strich)
    M_moon[1] = body[0] * body[2] * (A_B_strich - C_strich)
    M_moon[2] = body[0] * body[1] * (A_B_strich - A_B_strich)
    M_moon = scalar * M_moon
    
    # ___ sum
    m = np.zeros([3, ])
    m = M_moon + M_sun
#______________________________________________________________________________
    """ T_G(t) = sqrt(5/3)*M*R^2*(matrix) """
    scalar = np.sqrt(5 / 3) * M_earth * R_earth ** (2)
    #print("scalar = {}".format(scalar))
    
    matrix = np.zeros([3, 3])
    c_s = pc_aohis[index]
    c_s = c_s + pc_tide[index]
    # print("c_s = {}".format(c_s))
    matrix[0, 0] = (np.sqrt(1 / 3) * c_s[0]) - c_s[3]
    matrix[0, 1] = -c_s[4]
    matrix[0, 2] = -c_s[1]
    matrix[1, 0] = -c_s[4]
    matrix[1, 1] = (np.sqrt(1 / 3) * c_s[0]) + c_s[3]
    matrix[1, 2] = -c_s[2]
    matrix[2, 0] = -c_s[1]
    matrix[2, 1] = -c_s[2]
    matrix[2, 2] = -(2 * np.sqrt(1 / 3) * c_s[0])
    # print("matrix = \n{}".format(matrix))
    
    # tr_
    tr_ = A_B_strich + A_B_strich + C_strich
    #print("tr_ = {}".format(tr_))
    matrix_2 = np.eye(3)
    matrix_2 = matrix_2 * (tr_ / 3)
    #print("matrix_2 = \n{}".format(matrix_2))
    
    tg = (scalar * matrix) + matrix_2
    tg_v[index, :] = tg

#______________________________________________________________________________     
    """ T_R(t) = (O_N*R^5)/(3*G) * (matrix) """
    tr_scalar = (Omega_n * R_earth ** (5)) / (3 * G)
    #print("scalar = {}".format(scalar))

    # matrix
    tr_matrix = np.zeros([3, 3])
    w_x = w_v[index][0]
    w_y = w_v[index][1]

    tr_matrix[0, 2] = (k_Re * w_x + k_Im * w_y)
    tr_matrix[1, 2] = (k_Re * w_y - k_Im * w_x)
    tr_matrix[2, 0] = (k_Re * w_x + k_Im * w_y)
    tr_matrix[2, 1] = (k_Re * w_y - k_Im * w_x)
    #print("matrix = \n{}".format(matrix))

    tr = tr_scalar * tr_matrix
    tr_v[index, :] = tr

#______________________________________________________________________________       
    """ T(t) = T_G(t) + T_R(t) """
    t = tg_v[index, :] + tr_v[index, :]

#______________________________________________________________________________       
    # h
    h_x = aam[index, :] + oam[index, :] + ham[index, :]

    h = np.zeros([3, ])
    #h[0] = ((Omega_n * (C_strich - A_B_strich)) / 1.610) * h_x[3]
    #h[1] = ((Omega_n * (C_strich - A_B_strich)) / 1.610) * h_x[4]
    #h[2] = ((Omega_n *  C_strich) / 1.125) * h_x[5]
    
    h[0] = (1.610 / (Omega_n * (C_strich - A_B_strich))) * h_x[3]
    h[1] = (1.610 / (Omega_n * (C_strich - A_B_strich))) * h_x[4]
    h[2] = (1.125 / (Omega_n * C_strich)) * h_x[5]

    h_v[index, :] = h/3600
      
#______________________________________________________________________________
    """ dTG(t) = T_G(t) - T_R(t-1) """
    if index > 0:
        delta_t = tg_v[index] - tg_v[index - 1]
    #else:
    #    delta_t = tg_v[index] - 0
    delta_t = delta_t/3600
        

#______________________________________________________________________________
    """ dTG(t) = T_G(t) - T_R(t-1) """
    if index > 0:
        delta_h = h_v[index] - h_v[index - 1]
    else:
        delta_h = h_v[index] / 2
    delta_h = delta_h/3600
        
#______________________________________________________________________________
    """ force """
    f_scalar = (Omega_n**2 * R_earth ** (5)) / (3 * G)

    # matrix
    f_matrix = np.zeros([3, 3])
    f_matrix[0, 0] = k_Re
    f_matrix[0, 1] = k_Im
    f_matrix[1, 0] = -k_Im
    f_matrix[1, 1] = k_Re
    #print("matrix = \n{}".format(matrix))

    t_temp = f_scalar * f_matrix
    f = t + t_temp
#______________________________________________________________________________
    # w_dot
    
    w = w_v[index, :]
    """ F^(-1) """
    f_invers = np.linalg.inv(f)

    """ (D*T_G/Dt) * w """
    dT_G_Dt_w = np.dot(delta_t, w)
    #print('delta_t({}) = \n{}'.format(index, delta_t))

    """ w x (T * w) """
    tw = np.dot(t, w)
    w_x_tw = np.cross(w, tw)

    """ w x h(t) """
    w_x_h = np.cross(w, h_v[index, :] - delta_h)

    """ w_dot = F^(-1) * [M - ((D*T_G/Dt) * w) - (w x (T * w)) - (w x h) - (D_h/Dt)] """
    """ w_dot = F^(-1) * [M - (DT_G_Dt_w)      - (w_x_Tw)      - (w x h) - (D_h/Dt)] """
    w_dot = np.dot(f_invers, (m - dT_G_Dt_w - w_x_tw - w_x_h ))
    w_dot_v[index, :] = w_dot
    #print('w_dot({}) = \n{}'.format(index, w_dot_v[index, :]))
    
    w = w + w_dot * 3600
    w_v[index + 1, :] = w
    #print('w_v({}) = \n{}'.format(index, w_v[index, :]))

#______________________________________________________________________________
    # polar_motion
    """ x_p(t) = (R/W_N) * w_x(t), y_p(t) = (R/W_N) * w_y(t) """
    w = w_v[index]
    
    x_p = (R_earth / Omega_n) * w[0]
    y_p = (R_earth / Omega_n) * w[1]
    polar_v[index][0] = x_p
    polar_v[index][1] = y_p
    
    # polar_motion ref
    w = earth_rot[index]
    polar_v_ref[index][0] = (R_earth / Omega_n) * w[0]
    polar_v_ref[index][1] = (R_earth / Omega_n) * w[1]
    
 
#______________________________________________________________________________
    # delta_lod
    delta_lod_v[index] = 86400 * ((Omega_n - w_v[index][2]) / Omega_n)
    
    # delta_lod ref    
    delta_lod_v_ref[index] = 86400 * ((Omega_n - earth_rot[index][2]) / Omega_n)

# Don't touch it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
########################################################################################################################



#fileextention = '.pdf'
fileextention = '.png'
transp = True
# w
polar_lim_x = (-0.5 * 10**-10, 1.2 * 10**-10)
polar_lim_y = (-2 * 10**-10, -0.2 * 10**-10)
polar_lim_z = (2005, 2015)
line_width_polar = 0.5
w_v = w_v[1:len(w_v)-2]
fig = plt.figure(1)
ax = plt.axes(projection='3d', xlim=polar_lim_x, ylim=polar_lim_y, zlim=polar_lim_z)
ax.plot3D(w_v[:, 0], w_v[:, 1], np.linspace(start=2005, stop=2015, num=len(w_v)), 'red', lw=line_width_polar)
ax.set_xlabel("$\omega_x$ [rad/s * $10^{-10}$]" ,fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("$\omega_y$ [rad/s * $10^{-10}$]" ,fontsize=15, fontweight="bold", labelpad=10)
ax.set_zlabel("time [years]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('Simulated earth rotation', fontsize=19, fontweight="bold")
ax.view_init(elev=25, azim=0)
fig.savefig("earth_rotation"+fileextention, bbox_inches='tight', dpi=800, transparent=transp)

# w ref
earth_rot = earth_rot[1:len(earth_rot)-2]
fig = plt.figure(2)
ax = plt.axes(projection='3d', xlim=polar_lim_x, ylim=polar_lim_y, zlim=polar_lim_z)
ax.plot3D(earth_rot[:, 0], earth_rot[:, 1], np.linspace(start=2005, stop=2015, num=len(earth_rot)), 'blue', lw=line_width_polar)
ax.set_xlabel("$\omega_x$ [rad/s * $10^{-10}$]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("$\omega_y$ [rad/s * $10^{-10}$]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_zlabel("time [years]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('ref earth rotation', fontsize=19, fontweight="bold")
ax.view_init(elev=25, azim=0)
fig.savefig("earth_rotation_ref"+fileextention, bbox_inches='tight', dpi=800)

# w & w ref
fig = plt.figure(3)
ax = plt.axes(projection='3d', xlim=polar_lim_x, ylim=polar_lim_y, zlim=polar_lim_z)
ax.plot3D(w_v[:, 0], w_v[:, 1], np.linspace(start=2005, stop=2015, num=len(w_v)), 'red', lw=line_width_polar, label='sim')
ax.plot3D(earth_rot[:, 0], earth_rot[:, 1], np.linspace(start=2005, stop=2015, num=len(w_v)), 'blue', lw=line_width_polar, label='ref')
ax.set_xlabel("$\omega_x$ [rad/s * $10^{-10}$]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("$\omega_y$ [rad/s * $10^{-10}$]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_zlabel("time [years]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('Comparison Simulated- and ref earth rotation', fontsize=19, fontweight="bold")
#plt.legend(loc=0)
ax.view_init(elev=25, azim=0)
fig.savefig("comparison_simulated_earth_rotation_with_ref_e{}_a{}{}".format(25, 0, '.png'), dpi=800)
#ax.set_zlabel("", fontsize=15, fontweight="bold", labelpad=10)
#ax.view_init(elev=90, azim=90)
#fig.savefig("comparison_simulated_earth_rotation_with_ref_e{}_a{}{}".format(90, 90, '.png'), dpi=800)



if special_computation:
    # w & w g_half
    w_v_g_half = w_v_g_half[1:len(w_v_g_half)-2]
    fig = plt.figure(4)
    ax = plt.axes(projection='3d', zlim=polar_lim_z)
    ax.plot3D(w_v[:, 0], w_v[:, 1], np.linspace(start=2005, stop=2015, num=len(w_v)), 'red', lw=line_width_polar, label='Gravitational constant')
    ax.plot3D(w_v_g_half[:, 0], w_v_g_half[:, 1], np.linspace(start=2005, stop=2015, num=len(w_v)), 'green', lw=line_width_polar, label='Gravitational constant/2')
    ax.set_xlabel("$\omega_x$ [rad/s * $10^{-10}$]", fontsize=15, fontweight="bold", labelpad=30)
    ax.set_ylabel("$\omega_y$ [rad/s * $10^{-10}$]", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_zlabel("time [years]", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_title('Comparison simulated earth rotation with normal and half G', fontsize=19, fontweight="bold")
    #plt.legend(loc=0)
    fig.savefig("comparison_earth_rotation_with_normal_and_half_g.pdf", bbox_inches='tight')

    ax.view_init(elev=25, azim=0)
    fig.savefig("comparison_earth_rotation_with_normal_and_half_g_e{}_a{}{}".format(25, 0, '.png'), dpi=800)

    ax.set_zlabel("", fontsize=15, fontweight="bold", labelpad=10)
    ax.view_init(elev=90, azim=90)
    fig.savefig("comparison_earth_rotation_with_normal_and_half_g_e{}_a{}{}".format(90, 90, '.png'), dpi=800)
    '''
    for e in range(0, 95, 5):
        for a in range(0, 95, 5):
            ax.view_init(elev=e, azim=a)
            fig.savefig("movie/movie_e{}_a{}{}".format(e, a, '.png'), dpi=200)
    '''


# polar
polar_m_lim_x = (-6, 10)
polar_m_lim_y = (-18, -2)
polar_v = polar_v[1:len(polar_v)-2]
fig = plt.figure(5)
ax = plt.axes(xlim=polar_m_lim_x, ylim=polar_m_lim_y)
ax.plot(polar_v[:, 0], polar_v[:, 1], 'red', lw=line_width_polar)
ax.set_xlabel("$x_p$ [m]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("$y_p$ [m]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('Simulated polar motion', fontsize=19, fontweight="bold")
fig.savefig("polar_motion"+fileextention, bbox_inches='tight', dpi=800)

# polar ref
polar_v_ref = polar_v_ref[1:len(polar_v_ref)-2]
fig = plt.figure(6)
ax = plt.axes(xlim=polar_m_lim_x, ylim=polar_m_lim_y)
ax.plot(polar_v_ref[:, 0], polar_v_ref[:, 1], 'blue', lw=line_width_polar)
ax.set_xlabel("$x_p$ [m]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_ylabel("$y_p$ [m]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('ref polar motion', fontsize=19, fontweight="bold")
fig.savefig("polar_motion_ref"+fileextention, bbox_inches='tight', dpi=800)


# polar x & polar x ref
polar_m_lim_x_comp = (-10, 15)
polar_m_lim_y_comp = (-20, 0)
fig = plt.figure(7)
ax = plt.axes(xlim=polar_lim_z, ylim=polar_m_lim_x_comp)
ax.plot(np.linspace(start=2005, stop=2015, num=len(polar_v)), polar_v[:, 0], 'red', lw=line_width_polar, label='sim')
ax.plot(np.linspace(start=2005, stop=2015, num=len(polar_v)), polar_v_ref[:, 0], 'blue', lw=line_width_polar, label='ref')
ax.set_xlabel("time [years]", fontsize=15, fontweight="bold", labelpad=5)
ax.set_ylabel("$x_p$ [m]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('polar motion comparison x', fontsize=19, fontweight="bold")
plt.legend(loc=0)
plt.grid()
fig.savefig("polar_motion_comparison_x"+fileextention, bbox_inches='tight', dpi=800)

# polar y & polar y ref
fig = plt.figure(8)
ax = plt.axes(xlim=polar_lim_z, ylim=polar_m_lim_y_comp)
ax.plot(np.linspace(start=2005, stop=2015, num=len(polar_v)), polar_v[:, 1], 'red', lw=line_width_polar, label='sim')
ax.plot(np.linspace(start=2005, stop=2015, num=len(polar_v)), polar_v_ref[:, 1], 'blue', lw=line_width_polar, label='ref')
ax.set_xlabel("time [years]", fontsize=15, fontweight="bold", labelpad=5)
ax.set_ylabel("$y_p$ [m]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('polar motion comparison y', fontsize=19, fontweight="bold")
plt.legend(loc=0)
plt.grid()
fig.savefig("polar_motion_comparison_y"+fileextention, bbox_inches='tight', dpi=800)


# delta_lod
lod_lim = 0.003
line_width = 0.5
delta_lod_v = delta_lod_v[1:len(delta_lod_v)-2]
fig = plt.figure(10)
ax = plt.axes(ylim=(-lod_lim, lod_lim))
ax.plot(np.linspace(start=2005, stop=2015, num=len(delta_lod_v)), delta_lod_v, 'red', lw=line_width)
ax.set_xlabel("time [years]", fontsize=15, fontweight="bold", labelpad=5)
ax.set_ylabel("$\Delta$ LOD [s]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('Simulated $\Delta$ LOD', fontsize=19, fontweight="bold")
plt.grid()
fig.savefig("delta_lod_sim"+fileextention, bbox_inches='tight', dpi=800)

# delta_lod ref
delta_lod_v_ref = delta_lod_v_ref[1:len(delta_lod_v_ref)-2]
fig = plt.figure(11)
ax = plt.axes(ylim=(-lod_lim, lod_lim))
ax.plot(np.linspace(start=2005, stop=2015, num=len(delta_lod_v_ref)), delta_lod_v_ref, 'blue', lw=line_width)
ax.set_xlabel("time [years]", fontsize=15, fontweight="bold", labelpad=5)
ax.set_ylabel("$\Delta$ LOD [s]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('$\Delta$ LOD', fontsize=19, fontweight="bold")
plt.grid()
fig.savefig("delta_lod_ref"+fileextention, bbox_inches='tight', dpi=800)

# delta lod ref - delta lod
delta_delta_lod = delta_lod_v_ref - delta_lod_v
fig = plt.figure(12)
ax = plt.axes(ylim=(-lod_lim, lod_lim))
ax.plot(np.linspace(start=2005, stop=2015, num=len(delta_delta_lod)), delta_delta_lod, 'green', lw=line_width)
ax.set_xlabel("time [years]", fontsize=15, fontweight="bold", labelpad=5)
ax.set_ylabel("$\Delta$ LOD [s]", fontsize=15, fontweight="bold", labelpad=10)
ax.set_title('ref $\Delta$ $LOD_{Ref}$ - $\Delta$ $LOD_{Model}$', fontsize=19, fontweight="bold")
plt.grid()
fig.savefig("delta_lod_ref_minus_sim"+fileextention, bbox_inches='tight', dpi=800)

plt.show()
