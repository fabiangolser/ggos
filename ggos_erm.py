"""
GGOS Lab

This is the class represent the rotation model of the earth.
All computation is done in this class.
"""
import numpy as np
from numpy.core.multiarray import ndarray

import GGOS_pro1 as g_data


class RotationModel:
    def __init__(self, data):
        self.__data = data
        self.__index = 0
        self.__tg_current = np.zeros([3, 3])
        self.__tg_last = np.zeros([3, 3])
        self.__h_current = np.zeros([3, ])
        self.__h_last = np.zeros([3, ])

    def m_of_t(self, j):
        """ Fromel!!!!!!!!!! """
        M = np.zeros([3, ])

        if j == 1:  # sun
            body = self.__data.sun(self.__index)
            GM = self.__data.GM_sun
        if j == 2:  # moon
            body = self.__data.moon(self.__index)
            GM = self.__data.GM_moon

        r = np.sqrt(body[0]**2 + body[1]**2 + body[2]**2)
        scalar = 3 * GM / r**5
        M[0] = body[1] * body[2] * (self.__data.C_strich - self.__data.A_B_strich)
        M[1] = body[0] * body[2] * (self.__data.A_B_strich - self.__data.C_strich)
        M[2] = body[0] * body[1] * (self.__data.A_B_strich - self.__data.A_B_strich)

        return scalar * M

    def force(self):
        """ Fromel!!!!!!!!!! """
        scalar = (self.__data.Omega_n**2 * self.__data.R_earth ** (5)) / (3 * self.__data.G)
        #print("scalar = {}".format(scalar))

        # matrix
        matrix = np.zeros([3, 3])
        k_re = self.__data.k_Re
        k_im = self.__data.k_Im

        matrix[0, 0] = k_re
        matrix[0, 1] = k_im
        matrix[1, 0] = -k_im
        matrix[1, 1] = k_re
        #print("matrix = \n{}".format(matrix))

        return self.t_total() + scalar * matrix

    def tg_of_t(self):
        """ T_G(t) = sqrt(5/3)*M*R^2*(matrix) """
        # pre
        scalar = np.sqrt(5 / 3) * self.__data.M_earth * self.__data.R_earth ** (2)
        #print("scalar = {}".format(scalar))

        # matrix
        matrix = np.zeros([3, 3])
        c_s = self.__data.pc_aohis(self.__index)
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

        # tr
        tr = self.__data.A_B_strich + self.__data.A_B_strich + self.__data.C_strich
        #print("tr = {}".format(tr))
        matrix_2 = np.eye(3)
        matrix_2 = matrix_2 * (tr / 3)
        #print("matrix_2 = \n{}".format(matrix_2))

        self.__tg_last = self.__tg_current
        self.__tg_current = (scalar * matrix) + matrix_2
        return self.__tg_current

    def tr_of_t(self):
        """ T_R(t) = (O_N*R^5)/(3*G) * (matrix) """
        # pre
        scalar = (self.__data.Omega_n * self.__data.R_earth ** (5)) / (3 * self.__data.G)
        #print("scalar = {}".format(scalar))

        # matrix
        matrix = np.zeros([3, 3])
        k_re = self.__data.k_Re
        k_im = self.__data.k_Im
        w_x = self.__data.w[self.__index][0]
        w_y = self.__data.w[self.__index][1]

        matrix[0, 2] = (k_re * w_x + k_im * w_y)
        matrix[1, 2] = (k_re * w_y - k_im * w_x)
        matrix[2, 0] = (k_re * w_x + k_im * w_y)
        matrix[2, 1] = (k_re * w_y - k_im * w_x)
        #print("matrix = \n{}".format(matrix))

        return scalar * matrix

    def t_total(self):
        """ T(t) = T_G(t) + T_R(t) """
        return self.__tg_current + self.tr_of_t()

    def delta_tg(self):
        """ dTG(t) = T_G(t) - T_R(t-1) """
        return self.__tg_current - self.__tg_last

    def h_it_self(self):
        h_aam = self.__data.aam(self.__index)
        h_aom = self.__data.aom(self.__index)
        h_ham = self.__data.ham(self.__index)
        h_x = h_aam + h_aom + h_ham

        h = np.zeros([3, ])
        #h_x[0] = (1.610 / (self.__data.Omega_n * (self.__data.C_strich - self.__data.A_B_strich))) * h[3]
        #h_x[1] = (1.610 / (self.__data.Omega_n * (self.__data.C_strich - self.__data.A_B_strich))) * h[4]
        #h_x[2] = (1.125 / (self.__data.Omega_n * self.__data.C_strich)) * h[5]

        h[0] = ((self.__data.Omega_n * (self.__data.C_strich - self.__data.A_B_strich)) / 1.610) * h_x[3]
        h[1] = ((self.__data.Omega_n * (self.__data.C_strich - self.__data.A_B_strich)) / 1.610) * h_x[4]
        h[2] = ((self.__data.Omega_n * self.__data.C_strich) / 1.125) * h_x[5]

        self.__h_last = self.__h_current
        self.__h_current = h
        return h

    def delta_h(self):
        """ Dh_dt(t) = h(t) - h(t-1) """
        return self.__h_current - self.__h_last

    def omega_dot(self, index, m=np.zeros([3, ]), dt=np.zeros([3, 3]), h=np.zeros([3, ]), dh=np.zeros([3, ])):
        self.__index = index

        w = self.__data.w[index]
        #print('w({}) = \n{}'.format(index, w))

        TG = self.tg_of_t()
        #print('TG({}) = \n{}'.format(index, TG))
        T = self.t_total()
        #print('T({}) = \n{}'.format(index, T))
        DT_G = self.delta_tg()
        #print('DT_G({}) = \n{}'.format(index, DT_G))

        """ F^(-1) """
        f = self.force()
        #print('Force({}) = \n{}'.format(index, f))
        f_invers = np.linalg.inv(f)

        """ M_j """
        M_1 = self.m_of_t(1)
        M_2 = self.m_of_t(2)
        M = M_1 + M_2
        print('M({}) = \n{}'.format(index, M))

        """ (D*T_G/Dt) * w """
        DT_G_Dt_w = np.dot(DT_G, np.transpose(w))
        #print('(D*T_G/Dt) * w({}) = \n{}'.format(index, DT_G_Dt_w))

        """ w x (T * w) """
        Tw = np.dot(T, np.transpose(w))
        w_x_Tw = np.transpose(np.cross(w, np.transpose(Tw)))
        #print('w x (T * w)({}) = \n{}'.format(index, w_x_Tw))

        """ w x h(t) """
        h = self.h_it_self()
        #print('h({}) = \n{}'.format(index, h))
        w_x_h = np.cross(w, h)
        #print('w x h({}) = \n{}'.format(index, w_x_h))

        """ Dh_dt(t) """
        dh = self.delta_h()
        #print('dh({}) = \n{}'.format(index, dh))

        """ w_dot = F^(-1) * [M - ((D*T_G/Dt) * w) - (w x (T * w)) - (w x h) - (D_h/Dt)] """
        """ w_dot = F^(-1) * [M - (DT_G_Dt_w)      - (w_x_Tw)      - (w x h) - (D_h/Dt)] """
        w_dot = np.dot(f_invers, (M - DT_G_Dt_w    -  w_x_Tw       -  w_x_h  -  dh))
        #print('w_dot({}) = \n{}'.format(index, w_dot))

        w_dot = np.transpose(w_dot)
        ww_dot = w + w_dot  # + 0.00000000000001
        self.__data.append_w([ww_dot])
        if self.__index == 0:
            self.__data.w_dot = w_dot
        else:
            self.__data.append_w_dot(w_dot)

        return [w_dot], f_invers, M, DT_G_Dt_w, w_x_Tw, w_x_h, dh, w

    def polar_motion(self, index, use_ref=False):
        """ x_p(t) = (R/W_N) * w_x(t), y_p(t) = (R/W_N) * w_y(t) """
        if use_ref:
            w = self.__data.earth_rotation(index)
        else:
            w = self.__data.w[index]

        #print("w = \n{}".format(w))
        x_p = (self.__data.R_earth / self.__data.Omega_n) * w[0]
        y_p = (self.__data.R_earth / self.__data.Omega_n) * w[1]

        return [[x_p, y_p]]

