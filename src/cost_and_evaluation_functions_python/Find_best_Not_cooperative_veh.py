import numpy as np
import Vehicle
import safety_distance as sd
import copy
import pvd_planner_util
import Polynomial_funktion
import post_pro


def no_cooperative_trajectory(ego_vehicle, vehicle_list, street, number_of_samples, path_info_list, sampling_range_t):
    """

    :type ego_vehicle: Vehicle.EgoVehicle
    :param ego_vehicle: list of sampled ego-vehicles
    :type vehicle_list: list[Vehicle.Vehicle]
    :param vehicle_list: list of predicted vehicles in scenario
    :return:
    """

    a_cost_array = ego_vehicle.trajectory_samples.cost_matrix[:, ego_vehicle.trajectory_samples.at_cost]
    v_cost_array = ego_vehicle.trajectory_samples.cost_matrix[:, ego_vehicle.trajectory_samples.v_cost]
    sd_cost_array = np.zeros_like(v_cost_array)
    d_sd_envelop = street.lanes[1].width / 2
    for i in range(number_of_samples):
        sd_cost = calculate_ego_veh_sd_cost(ego_vehicle, vehicle_list, i, d_sd_envelop)
        sd_cost_array[i] += sd_cost

    over_all_cost = a_cost_array + v_cost_array + sd_cost_array
    min_cost_i = np.argmin(over_all_cost)
    vehicle_list_new = copy.deepcopy(vehicle_list)
    create_best_cooperative_vehicle_list(ego_vehicle, vehicle_list_new, street, path_info_list, min_cost_i)
    print 'best no cooperative cost:', over_all_cost[min_cost_i]
    print 'a: ', a_cost_array[min_cost_i]
    print 'v: ', v_cost_array[min_cost_i]
    print 'sd: ', sd_cost_array[min_cost_i]
    post_pro.plot_scenario(vehicle_list_new, sampling_range_t, ego_vehicle.trajectory_samples.t_list, street, 0.01,
                           '/home/diestmann/Documents/post_pro/pyplot2/')


def calculate_ego_veh_sd_cost(ego_vehicle, vehicle_list, ego_i, d_sd_envelop):
    tra_samples = ego_vehicle.trajectory_samples
    cost = 0
    for j in range(len(vehicle_list)):
        cost += sd.sd_cost_for_ego(tra_samples.s_ref_matrix[ego_i], vehicle_list[j].trajectory.s_reference_list,
                                  tra_samples.d_matrix[ego_i], vehicle_list[j].trajectory.d_list,
                                  tra_samples.v_matrix[ego_i], ego_vehicle, vehicle_list[j], d_sd_envelop,
                                  ego_vehicle.cost_func)
        cost += sd.sd_cost_for_ego(vehicle_list[j].trajectory.s_reference_list, tra_samples.s_ref_matrix[ego_i], vehicle_list[j].trajectory.d_list,
                                   tra_samples.d_matrix[ego_i], vehicle_list[j].trajectory.v_list, vehicle_list[j], ego_vehicle, d_sd_envelop, vehicle_list[j].cost_func)

    return cost


def create_best_cooperative_vehicle_list(ego_vehicle, vehicle_list, street, path_info_list, index_opt):
    path_info_list[index_opt].edge.append(ego_vehicle.trajectory_samples.s_ref_matrix[index_opt, -1] + 1.)
    path = Polynomial_funktion.PiecewiseDefinedFunction(path_info_list[index_opt].segments,
                                                        path_info_list[index_opt].edge).value

    ego_trajectory = pvd_planner_util.TrajectoryNew(ego_vehicle.trajectory_samples.v_matrix[index_opt, :],
                                                    ego_vehicle.trajectory_samples.at_matrix[index_opt, :],
                                                    ego_vehicle.trajectory_samples.s_ref_matrix[index_opt, :],
                                                    ego_vehicle.trajectory_samples.d_matrix[index_opt, :],
                                                    ego_vehicle.trajectory_samples.lane_matrix[index_opt, :],
                                                    street, path)

    ego_vehicle.trajectory = ego_trajectory
    vehicle_list.append(ego_vehicle)
