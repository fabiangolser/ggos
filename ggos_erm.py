"""
GGOS Lab

This is the class represent the rotation model of the earth.
All computation is done in this class.
"""
import numpy as np
import GGOS_pro1 as g_data


class RotationModel:

    def __init__(self, data):
        self.__data = data
        self.__index = 0

    def m_of_t(self, j):
        """ Fromel!!!!!!!!!! """
        return np.zeros([3, ])

    def force(self, j):
        """ Fromel!!!!!!!!!! """
        pass

    def tg_of_t(self):
        """ T_G(t) = sqrt(5/3)*M*R^2*(matrix) """
        # pre
        scalar = np.sqrt(5 / 3) * self.__data.M_earth * self.__data.R_earth**(2)
        #print("scalar = {}".format(scalar))

        # matrix
        matrix = np.zeros([3, 3])
        c_s = self.__data.pc_aohis(self.__index)
        #print("c_s = {}".format(c_s))
        matrix[0, 0] = (np.sqrt(1/3)*c_s[0]) - c_s[3]
        matrix[0, 1] = -c_s[4]
        matrix[0, 2] = -c_s[1]
        matrix[1, 0] = -c_s[4]
        matrix[1, 1] = (np.sqrt(1/3)*c_s[0]) + c_s[3]
        matrix[1, 2] = -c_s[2]
        matrix[2, 0] = -c_s[1]
        matrix[2, 1] = -c_s[2]
        matrix[2, 2] = -(2*np.sqrt(1/3)*c_s[0])
        #print("matrix = \n{}".format(matrix))

        # tr
        tr = self.__data.A_B_strich + self.__data.C_strich
        #print("tr = {}".format(tr))
        matrix_2 = np.eye(3)
        matrix_2 = matrix_2 * (tr / 3)
        #print("matrix_2 = \n{}".format(matrix_2))

        return (scalar * matrix) + matrix_2

    def tr_of_t(self):
        """ T_R(t) = (O_N*R^5)/(3*G) * (matrix) """
        # pre
        scalar = (self.__data.Omega_n * self.__data.R_earth**(5)) / (3 * self.__data.G)
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
        return self.tg_of_t() + self.tr_of_t()

    def h_it_self(self):
        pass

    def delta_t(self):
        pass

    def delta_h(self):
        pass

    def omega_dot(self, index, m=np.zeros([3, ]), dt=np.zeros([3, 3]), h=np.zeros([3, ]), dh=np.zeros([3, ])):
        self.__index = index
        w_dot = np.array([0, 0, 0])

        t_total = self.t_total()
        print('t_total = \n{}'.format(t_total))

        self.__data.append_w_dot(w_dot)
        print('w_dot = \n{}'.format(self.__data.w_dot))
        return w_dot
