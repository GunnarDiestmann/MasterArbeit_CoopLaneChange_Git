import bisect
import numpy as np


class Poly5(object):
    """
    Polynomial 5th order function for given start and end values
    """
    def __init__(self, x_start, y_start, x_end, y_end, x_start_ab1, x_start_ab2, x_end_ab1, x_end_ab2):
        """
        :param x_start: x_0 value
        :param y_start: y(x_0) value
        :param x_end: x_end value
        :param y_end: y(x_end) Value
        :param x_start_ab1: y(x_0) first order derivation
        :param x_start_ab2: y(x_0) second order derivation
        :param x_end_ab1: y(x_end) first order derivation
        :param x_end_ab2: y(x_end) second order derivation
        """
        self.x_start = x_start
        self.k_0 = y_start
        self.k_1 = x_start_ab1
        self.k_2 = (x_start_ab2/2)
        self.k_5 = (12 * (y_end - y_start) - 6 * x_start_ab1 * (x_end - x_start) - 6 * x_end_ab1 * (
                    x_end - x_start) - 2 * x_start_ab2 * (x_end - x_start) ** 2 + x_end_ab2 * (
                                x_end - x_start) ** 2) / (2 * (x_end - x_start) ** 5)
        self.k_4 = (x_end_ab1 * (x_end - x_start) + 2 * x_start_ab1 * (x_end - x_start) + x_start_ab2 * (
                    x_end - x_start) ** 2 - 3 * (y_end - y_start) - 2 * self.k_5 * (x_end - x_start) ** 5) / (
                               (x_end - x_start) ** 4)
        self.k_3 = ((y_end - y_start) - x_start_ab1 * (x_end - x_start) - x_start_ab2 * (x_end - x_start) - self.k_5 * (
                    x_end - x_start) ** 5 - self.k_4 * (x_end - x_start) ** 4) / ((x_end - x_start) ** 3)

    def value(self, x):
        return self.k_0 + self.k_1 * (x - self.x_start) + self.k_2 * (x - self.x_start) ** 2 + self.k_3 * (
                    x - self.x_start) ** 3 + self.k_4 * (x - self.x_start) ** 4 + self.k_5 * (x - self.x_start) ** 5


class Ploy4(object):
    def __init__(self, x1, y1, x2, y2, x3, y3, y1_d, y3_d):
        self.x_start = x1
        self.k_0 = y1
        self.k_1 = y1_d

        print y3_d
        print x3

        a11 = y2 - self.k_0 - self.k_1*x2
        a12 = x2**2.
        print a12
        a13 = x2**3.
        a14 = x2**4.
        a21 = y3 - self.k_0 - self.k_1*x3
        a22 = x3**2.
        a23 = x3**3.
        a24 = x3**4.
        a31 = y3_d - self.k_1
        a32 = 2.*x3
        print a32
        a33 = 3.*(x3**2.)
        a34 = 4.*(x3**3.)

        a1 = a22/a12
        a2 = a32/a12
        print a2

        b11 = a21 - a11*a1
        b12 = a23 - a13*a1
        b13 = a24 - a14*a1

        b21 = a31 - a11 * a2
        b22 = a33 - a13 * a2
        b23 = a34 - a14 * a2

        self.k_4 = (b21 - b11*(b22/b12)) / (b23 - b13*(b22/b12))

        self.k_3 = (b11 - b13*self.k_4) / b12

        self.k_2 = (a11-a13*self.k_3-a14*self.k_4) / a12

    def value(self, x):
        return self.k_0+self.k_1*x+self.k_2*x**2.+self.k_3*x**3.+self.k_4*x**4.

    def ab_value(self, x):
        return self.k_1+2.*self.k_2*x+3.*self.k_3*x**2.+4.*self.k_4*x**3.


class PiecewiseDefinedFunction(object):
    """
    This class represents a piecewise defined function.
    """
    def __init__(self, function_list, edge_list):
        """
        Init the piecewise function
        :type function_list: list[function]
        :param function_list: List with all functions of the piecewise defined function, sorted rising
        :type edge_list: list[float]
        :param edge_list: list with all edges of the piecewise defined function, sorted rising
        """
        self.function_list = function_list
        self.edge_list = edge_list

    def value(self, s):
        """
        Returns the value of the piecewise function at the given point s
        :type s: float
        :param s: point at which the value of the function is required
        :return: value of the piecewise function at s
        """
        function_index = np.searchsorted(self.edge_list, s)-1
        try:
            value_list = []
            for func, s_singel in zip(function_index, s):
                try:
                    value_list.append(self.function_list[func](s_singel))
                except IndexError:
                    print 'max s', self.edge_list[-1]
                    print 's= ', s
                    raise ValueError
            return value_list
        except TypeError:
            return self.function_list[function_index](s)
