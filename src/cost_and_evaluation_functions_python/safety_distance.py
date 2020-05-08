import numpy as np
import Vehicle
import post_pro


def safety_distance_envelop(v, alpha, lane_width=3.):
    """
    Calculate the distance from the vehicle coordinate system to the safety distance envelop for a given angle
    :param v: velocity
    :param alpha: angle for distance to safety distance envelop
    :param lane_width: width of lane
    :return:
    """
    a = v*3.6*0.5
    b = lane_width/2.
    alpha1 = np.arctan(np.tan(alpha)*(b/a))
    x = a*np.sin(alpha1)
    y = b*np.cos(alpha1)
    return (x**2.+y**2.)**0.5


def calc_sd_array(veh1, veh2, s_sub, d_sub, v_veh1):
    """
    Calculate the normed distance -> distance/required_safety_distance
    :type veh1: Vehicle.Vehicle
    :param veh1: vehicle for which the safety distance is to be checked
    :type veh2: Vehicle.Vehicle
    :param veh2: preceding vehicle
    :param s_sub: s_reference_veh2 - s_reference_veh1
    :param d_sub: d(s_reference)_veh2 - d(s_reference_veh1)
    :param v_veh1: velocity list of veh1
    :return: normed distance
    """
    s_distance_m = s_sub - veh1.properties.length_to_rear_bumper
    s_distance_m[s_distance_m < 0.] = 0.

    d_distance_m = d_sub - veh2.properties.width/2
    d_distance_m[d_distance_m < 0.] = 0.00001

    alpha_m = np.arctan(s_distance_m/d_distance_m)
    distance_m = (s_distance_m**2. + d_distance_m**2.)**0.5
    veh1_env_m = veh1.envelop.value(alpha_m)

    sd_env_m = safety_distance_envelop(v_veh1, alpha_m)
    normed_distance_m = (distance_m - veh1_env_m) / (sd_env_m - veh1_env_m)
    return normed_distance_m


def calc_sd_cost(cost_func, distance_list, mask):
    cost_matrix = np.zeros_like(mask, dtype=float)
    cost_array_mask = cost_func.calc_sd_costs_for_mask(distance_list)
    cost_matrix[mask] = cost_array_mask
    cost_array = np.sum(cost_matrix, axis=1)
    return cost_array


def calc_sd_cost_value(cost_func, distance_list, mask):
    cost_matrix = np.zeros_like(mask, dtype=float)
    cost_array_mask = cost_func.calc_sd_costs_for_mask(distance_list)
    cost_matrix[mask] = cost_array_mask
    cost_array = np.sum(cost_matrix)
    return cost_array


def calc_normed_distance_mask(veh_s, prec_veh_s, veh_d, prec_veh_d, veh_v, veh, prec_veh, max_sd_envelop_d=3.5/2.):
    s_sub = prec_veh_s - veh_s
    d_sub = np.absolute(veh_d - prec_veh_d)
    veh1_v_matrix = np.where(veh_v < 5., 5., veh_v)

    max_sd_envelop_s = np.max(veh1_v_matrix) * 3.6 * 0.5
    max_required_s_sub = max_sd_envelop_s + prec_veh.properties.length_to_rear_bumper
    # sort out all time stamps where the other vehicle is behind me
    s_sub_mask_1 = s_sub > 0.
    # sort our all time stamps where the other vehicle can't be in the safety distance area
    s_sub_mask_2 = s_sub < max_required_s_sub

    # sort our all time stamps where the other vehicle is on other lane
    max_required_d_sub = max_sd_envelop_d + prec_veh.properties.width / 2
    d_sub_mask = d_sub < max_required_d_sub

    combined_mask = s_sub_mask_1 & s_sub_mask_2 & d_sub_mask

    # Use just the time stamps that pass the 3 masks
    s_sub_masked = s_sub[combined_mask]
    d_sub_masked = d_sub[combined_mask]
    veh1_v_matrix_masked = veh1_v_matrix[combined_mask]

    normed_distance_m = calc_sd_array(veh, prec_veh, s_sub_masked, d_sub_masked, veh1_v_matrix_masked)

    return normed_distance_m, combined_mask


def sd_cost_samples_to_veh(veh_s_matrix, prec_veh_s_array, veh_d_matrix, prec_veh_d_array, veh_v_matrix, veh, prec_veh,
                           cost_func, max_sd_envelop_d=3.5/2.):
    normed_distance_m, combined_mask = calc_normed_distance_mask(veh_s_matrix, prec_veh_s_array, veh_d_matrix,
                                                                 prec_veh_d_array, veh_v_matrix, veh, prec_veh,
                                                                 max_sd_envelop_d)
    cost = calc_sd_cost(cost_func, normed_distance_m, combined_mask)
    return cost


def sd_costs_veh_to_ego(veh_s, samples_s, veh_d, samples_d, veh_v, veh, sample_veh, cost_func, max_sd_envelop_d=3.5/2.):
    veh_v_matrix = np.empty_like(samples_s)
    number_of_samples = np.shape(samples_s)[1]
    for i in range(number_of_samples):
        veh_v_matrix[i, :] = veh_v

    cost = sd_cost_samples_to_veh(veh_s, samples_s, veh_d, samples_d, veh_v_matrix, veh, sample_veh, cost_func,
                                  max_sd_envelop_d)

    return cost


def sd_cost_veh_to_veh(veh_s, prec_veh_s, veh_d, prec_veh_d, veh_v, veh, prec_veh, cost_func, max_sd_envelop_d=3.5/2.):
    normed_distance_m, combined_mask = calc_normed_distance_mask(veh_s, prec_veh_s, veh_d, prec_veh_d, veh_v, veh,
                                                                 prec_veh, max_sd_envelop_d)

    cost = calc_sd_cost_value(cost_func, normed_distance_m, combined_mask)
    # post_pro.unit_test_sd_calc(veh1_s_matrix, veh2_s_array, veh1_d_matrix, veh2_d_array, veh1_v_matrix, veh1, veh2, cost_func, combined_mask, normed_distance_m)
    return cost
