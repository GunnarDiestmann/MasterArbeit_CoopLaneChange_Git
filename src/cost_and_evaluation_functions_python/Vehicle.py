import numpy as np
import cost_function


class VehicleList(object):
    veh_list = []
    num_of_veh_tl = 0
    num_of_veh_il = 0


class VehicleEnvelop(object):
    """
    This class describes the outlines of a vehicle
    """
    def __init__(self, veh_properties):
        """
        :type veh_properties: VehicleProperties
        :param veh_properties: dimensional properties of a vehicle
        """
        self.alpha1 = np.arctan(-veh_properties.length_to_rear_bumper / (veh_properties.width / 2))
        self.alpha2 = np.arctan((1. / (veh_properties.width / 2)))
        self.a = -veh_properties.length_to_rear_bumper
        self.b = (veh_properties.width / 2)

    def value(self, alpha):
        """
        :type alpha: double
        :param alpha: angle at which the distance to the vehicle envelop needs to be determined
        :return: distance from vehicle coordinate system to vehicle envelop
        """

        return (self.a/(np.sin(alpha)+0.000001))*(alpha < self.alpha1) + \
               (self.b/(np.cos(alpha) + 0.000001))*(alpha >= self.alpha1)*(alpha < self.alpha2) + \
               (1. / (np.sin(alpha) + 0.000001)) * (alpha >= self.alpha2)


class VehicleProperties(object):
    """
    This Class represents all properties of a vehicle
    """
    def __init__(self, length, width):
        """
        :param length: length of the vehicle
        :param width: width of the vehicle
        """
        self.length = length  # length of the vehicle
        self.width = width  # width of the vehicle
        self.length_to_front_bumper = 1.  # position of the coordination system measured form the front bumper
        self.length_to_rear_bumper = self.length - self.length_to_front_bumper


class Vehicle(object):
    """
    This class represents vehicles. A vehicle is described by it length and width, the position of the coordinate system
    and by a list of possible trajectories. When the trajectory is given this list will have just one object
    """
    def __init__(self, veh_index, s_start, d_start, v_start, v_opt, v_max, dt, a_start=0., length=5., width=2.):
        """
        :type veh_index: int
        :param veh_index: index of the vehicle
        :type s_start: float
        :param s_start: starting position of the vehicle
        :type v_start: float
        :param v_start: speed at time t
        :type a_start: float
        :param a_start: acceleration at time t
        :type length: float
        :param length: length of the vehicle
        :type width: float
        :param width: vehicle width
        """

        self.veh_index = veh_index
        self.properties = VehicleProperties(length, width)
        self.envelop = VehicleEnvelop(self.properties)
        self.s_start = s_start
        self.d_start = d_start
        self.v_start = v_start
        self.v_opt = v_opt
        self.v_max = v_max
        self.a_start = a_start

        self.path = None

        self.cost_func = cost_function.CostFunction(dt)
        self.cost_func.v_opt = v_opt
        self.cost_func.reset_eval_functions()
