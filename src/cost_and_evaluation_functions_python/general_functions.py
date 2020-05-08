import numpy as np


# lane related functions
def determine_new_lane(veh_trajectory, veh, street, t):
    """
    :type veh_trajectory: pvd_planner_util.Trajectory
    :param veh_trajectory: trajectory of the vehicle
    :type veh: Vehicle.Vehicle
    :param veh: vehicle
    :type street: Street.Street
    :param street: road segment
    :type t: float
    :param t: time op lane change
    :return:
    """
    s = veh_trajectory.s_reference_list[int((t-veh_trajectory.t_start)/veh_trajectory.sampling_range_t)]
    d_veh = veh_trajectory.path(s)

    lanes_at_t = []
    for lane in street.lanes:
        if check_if_on_lane(lane, d_veh, s, street, veh.properties):
            lanes_at_t.append(lane)

    s_prev = veh_trajectory.s_reference_list[int((t - veh_trajectory.sampling_range_t - veh_trajectory.t_start) /
                                                 veh_trajectory.sampling_range_t)]
    d_veh_prev = veh_trajectory.path(s_prev)
    previous_lane = None
    for lane in street.lanes:
        if check_if_on_lane(lane, d_veh_prev, s_prev, street, veh.properties):
            previous_lane = lane
            break

    if previous_lane == lanes_at_t[0]:
        return lanes_at_t[1]
    else:
        return lanes_at_t[0]


def check_if_on_lane(lane, d_veh, s, street, veh_properties):
    """
    This function gives back all lanes in lane_list on the veh is currently driving on
    :type lane: int
    :param lane: lane_index
    :type d_veh: float
    :param d_veh: distance of the current veh path to the reference curve
    :type s: float
    :param s: current s_reference of the vehicle
    :type street: Street.Street
    :param street: road segment
    :type veh_properties: Vehicle.VehicleProperties
    :param veh_properties: dimensions of the vehicle
    :return: Boolean
        """
    d_lane = street.lanes[lane].value(s)
    veh_distance_to_lane_middle = abs(d_veh-d_lane)
    if veh_distance_to_lane_middle < ((street.lanes[lane].width/2)+(veh_properties.width/2)):
        return True

    return False


def check_if_veh_on_lane_at_t(lane, veh, t_key):
    """
    :type lane: int
    :param lane: lane index
    :type veh: Vehicle.Vehicle
    :type t_key: int
    :param t_key: key of the trajectory list of the time where it needs to be checked if the vehicle is on the lane
    :return:
    """
    street = veh.trajectory.street
    s = veh.trajectory.s_reference_list[t_key]
    d_veh = veh.trajectory.d_list[t_key]

    d_lane = street.lanes[lane].value(s)
    veh_distance_to_lane_middle = abs(d_veh-d_lane)
    if veh_distance_to_lane_middle < ((street.lanes[lane].width/2)+(veh.properties.width/2)):
        return True

    return False


# Frenet coordinate system related functions
def frenet_to_world_coordinates(x_reference, y_reference, s, d):
    """
    This function will convert a point given in frenet coordination system into global coordination system
    :type x_reference: function
    :param x_reference: x(s)
    :type y_reference function
    :param y_reference: y(s)
    :type s: float
    :param s: way along reference curve
    :type d: float
    :param d: perpendicular distance to reference curve
    :return: x, y position
    """
    x_reference_d = (x_reference(s + 0.1) - x_reference(s - 0.1)) / (2 * 0.1)
    y_reference_d = (y_reference(s + 0.1) - y_reference(s - 0.1)) / (2 * 0.1)
    try:
        theta_r = np.arctan2(y_reference_d, x_reference_d)
    except RuntimeWarning:
        theta_r = np.pi / 2
    x = (x_reference(s) - np.sin(theta_r) * d)
    y = (y_reference(s) + np.cos(theta_r) * d)

    return [x, y]


def calc_s_next(s, x_reference, y_reference, path, delta_t, v, a):
    """
    This function returns the next s_reference when a vehicle is driving a long a path given in frenet coordinates d(s)
    :type s: float
    :param s: current position of the vehicle referred to the reference curve
    :type x_reference: function
    :param x_reference: reference curve of the road x(s)
    :type y_reference function
    :param y_reference: reference curve of the road y(s)
    :type path: function
    :param path: path of the vehicle given in frenet coordinates d(s) referring to reference curve
    :type delta_t: float
    :param delta_t: time step
    :type v: float
    :param v: speed of last time step
    :type a: float
    :param a: acceleration of last time step
    :return: next s
    """
    path_d = (path(s + 0.1) - path(s - 0.1)) / (2 * 0.1)
    theta_path = np.arctan(path_d)

    x_reference_d = (x_reference(s + 0.1) - x_reference(s - 0.1)) / (2 * 0.1)
    y_reference_d = (y_reference(s + 0.1) - y_reference(s - 0.1)) / (2 * 0.1)
    x_reference_dd = (x_reference(s + 0.1) - (2 * x_reference(s)) + x_reference(s - 0.1)) / (0.1*0.1)
    y_reference_dd = (y_reference(s + 0.1) - (2 * y_reference(s)) + y_reference(s - 0.1)) / (0.1*0.1)

    curvature = calc_curvature(x_reference_d, y_reference_d, x_reference_dd, y_reference_dd)

    s_reference_next = s + (np.cos(theta_path) * calc_ds(delta_t, v, a)) / (1 - curvature * path(s))

    return s_reference_next


# Others
def calc_ds(delta_t, v0, a0):
    """
    This function calculates the distance s during a time period delta_t for a given velocity and acceleration
    :type delta_t: float
    :param delta_t: time step
    :type v0: float
    :param v0: speed at t = 0
    :type a0: float
    :param a0: acceleration at t = 0
    :return: delta_s
    """
    return v0 * delta_t + 0.5 * a0 * delta_t * delta_t


def calc_curvature(x_d, y_d, x_dd, y_dd):
    """
    Calculate curvature for a given first order and second order derivation of x and y
    :param x_d:
    :param y_d:
    :param x_dd:
    :param y_dd:
    :return:
    """
    try:
        return (x_d * y_dd - y_d * x_dd) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))
    except ZeroDivisionError:
        return 0
    except RuntimeWarning:
        return 0


def curvature_three_points(x1, x2, x3, y1, y2, y3):
    """
    Calculate the curvature by fitting a circle trough three points. Inverse of the circles radius is the curvature
    :param x1: x pos of point before
    :param x2: x pos at the point of the calculated curvature
    :param x3: x pos of the point behind
    :param y1: y pos of point before
    :param y2: y pos at the point of the calculated curvature
    :param y3: y pos of the point behind
    :return:
    """
    a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    b = (x1 ** 2 + y1 ** 2) * (y3 - y2) + (x2 ** 2 + y2 ** 2) * (y1 - y3) + (x3 ** 2 + y3 ** 2) * (y2 - y1)
    c = (x1 ** 2 + y1 ** 2) * (x2 - x3) + (x2 ** 2 + y2 ** 2) * (x3 - x1) + (x3 ** 2 + y3 ** 2) * (x1 - x2)
    d = (x1 ** 2 + y1 ** 2) * (x3 * y2 - x2 * y3) + (x2 ** 2 + y2 ** 2) * (x1 * y3 - x3 * y1) + (
            x3 ** 2 + y3 ** 2) * (x2 * y1 - x1 * y2)
    try:
        return 1 / (((b ** 2 + c ** 2 - 4 * a * d) / (4 * a ** 2)) ** 0.5)
    except RuntimeWarning:
        return 0


def calc_heading_angle(x_reference, y_reference, s):
    """
    Calaculate the heading angle of the reference curve
    :type x_reference: function
    :param x_reference: x(s) of the reference curve
    :type y_reference: function
    :param y_reference: y(s) of the reference curve
    :type s: float
    :param s: current position of the veh in frenet coordinates
    :return: heading angle
    """
    x_reference_d = (x_reference(s + 0.1) - x_reference(s - 0.1)) / (2 * 0.1)
    y_reference_d = (y_reference(s + 0.1) - y_reference(s - 0.1)) / (2 * 0.1)
    try:
        theta_r = np.arctan2(y_reference_d, x_reference_d)
    except RuntimeWarning:
        theta_r = np.pi / 2

    return theta_r
