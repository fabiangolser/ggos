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
        self.__t_total_current = np.zeros([3, 3])
        self.__t_total_last = np.zeros([3, 3])

    def m_of_t(self, j):
        """ Fromel!!!!!!!!!! """
#        if j ==1: #sun
#            const = ()/()
        return np.zeros([3, ])

    def force(self):
        """ Fromel!!!!!!!!!! """
        scalar = (self.__data.Omega_n**2 * self.__data.R_earth ** (5)) / (3 * self.__data.G)
        print("scalar = {}".format(scalar))

        # matrix
        matrix = np.zeros([3, 3])
        k_re = self.__data.k_Re
        k_im = self.__data.k_Im

        matrix[0, 0] = (k_re)
        matrix[0, 1] = (k_im)
        matrix[1, 0] = (-k_im)
        matrix[1, 1] = (k_re)
        print("matrix = \n{}".format(matrix))

        return self.t_total() + scalar * matrix

    def tg_of_t(self):
        """ T_G(t) = sqrt(5/3)*M*R^2*(matrix) """
        # pre
        scalar = np.sqrt(5 / 3) * self.__data.M_earth * self.__data.R_earth ** (2)
        #print("scalar = {}".format(scalar))

        # matrix
        matrix = np.zeros([3, 3])
        c_s = self.__data.pc_aohis(self.__index)
        #print("c_s = {}".format(c_s))
        matrix[0, 0] = (np.sqrt(1 / 3) * c_s[0]) - c_s[3]
        matrix[0, 1] = -c_s[4]
        matrix[0, 2] = -c_s[1]
        matrix[1, 0] = -c_s[4]
        matrix[1, 1] = (np.sqrt(1 / 3) * c_s[0]) + c_s[3]
        matrix[1, 2] = -c_s[2]
        matrix[2, 0] = -c_s[1]
        matrix[2, 1] = -c_s[2]
        matrix[2, 2] = -(2 * np.sqrt(1 / 3) * c_s[0])
        #print("matrix = \n{}".format(matrix))

        # tr
        tr = self.__data.A_B_strich + self.__data.C_strich
        # print("tr = {}".format(tr))
        matrix_2 = np.eye(3)
        matrix_2 = matrix_2 * (tr / 3)
        #print("matrix_2 = \n{}".format(matrix_2))

        return (scalar * matrix) + matrix_2

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
        self.__t_total_last = self.__t_total_current
        self.__t_total_current = self.tg_of_t() + self.tr_of_t()
        return self.__t_total_current

    def h_it_self(self):
        pass

    def delta_tg(self):
        delta_tg = self.__t_total_current - self.__t_total_last
        return delta_tg

    def delta_h(self):
        pass

    def omega_dot(self, index, m=np.zeros([3, ]), dt=np.zeros([3, 3]), h=np.zeros([3, ]), dh=np.zeros([3, ])):
        self.__index = index
        w_dot = np.array([0, 0, 0])

        w = self.__data.w
        #print('w({}) = \n{}'.format(index, w))
        DT_G = self.delta_tg()
        #print('DT_G({}) = \n{}'.format(index, DT_G))
        T = self.t_total()
        #print('T({}) = \n{}'.format(index, T))

        """ (D*T_G/Dt) * w """
        DT_G_Dt_w = np.dot(DT_G, np.transpose(w))
        #print('(D*T_G/Dt) * w({}) = \n{}'.format(index, DT_G_Dt_w))

        f = self.force()
        #print('Force({}) = \n{}'.format(index, f))
        f_invers = np.linalg.inv(f)
        """ w x (T * w) """
        Tw = np.dot(T, np.transpose(w))
        w_x_Tw = np.transpose(np.cross(w, np.transpose(Tw)))
        #print('w x (T * w)({}) = \n{}'.format(index, w_x_Tw))

        """ w_dot = F^(-1) * [M - ((D*T_G/Dt) * w) - (w x (T * w)) - (w x h) - (D_h/Dt)] """
        """ w_dot = F^(-1) * [M - (DT_G_Dt_w)      - (w_x_Tw)      - (w x h) - (D_h/Dt)] """
        w_dot     = np.dot(f_invers, (DT_G_Dt_w) - (w_x_Tw))
        #print('w_dot({}) = \n{}'.format(index, w_dot))

        w_dot = np.transpose(w_dot)
        if self.__index == 0:
            self.__data.w = (w + w_dot*1)
        else:
            self.__data.append_w_dot(w_dot)
            
        self.__data.append_w(w + w_dot*1)
        return w_dot

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

