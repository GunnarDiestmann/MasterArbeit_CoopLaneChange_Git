import pvd_planner_util
import numpy as np
import safety_distance as sd
import prediction
import post_pro


class CooperativePrediction(object):
    v = None
    a = None
    s_ref = None
    d = None
    vehicle_key = None

    def __init__(self, array_size, vehicle_key):
        self.vehicle_key = vehicle_key
        self.v = np.empty(array_size)
        self.a = np.empty(array_size)
        self.s_ref = np.empty(array_size)
        self.d = np.empty(array_size)

    def coop_trajectory_lc(self, vehicle_list, relevant_veh, v, a, s, d, num_of_samples, i_start_coop, i_end_coop, dt):
        self.v = v[self.vehicle_key]
        self.a = a[self.vehicle_key]
        self.s_ref = s[self.vehicle_key]
        self.d = d[self.vehicle_key]

        v_coop, a_coop, s_coop, d_coop = self.generate_coop_trajectory(vehicle_list, relevant_veh, v, a, s, d,
                                                                       i_start_coop, i_end_coop, dt, num_of_samples)

        self.v[i_start_coop:i_end_coop] = v_coop
        self.a[i_start_coop:i_end_coop] = a_coop
        self.s_ref[i_start_coop:i_end_coop] = s_coop
        self.d[i_start_coop:i_end_coop] = d_coop

        if i_end_coop < np.shape(v)[1]:
            v_idm, a_idm, s_idm, d_idm = self.idm_prediction(vehicle_list, i_end_coop-1, v, s, a, dt)
            self.v[i_end_coop-1:] = v_idm
            self.a[i_end_coop-1:] = a_idm
            self.s_ref[i_end_coop-1:] = s_idm
            self.d[i_end_coop-1:] = d_idm

    def generate_coop_trajectory(self, vehicle_list, relevant_veh, v, a, s, d, i_start_coop, i_end_coop, dt,
                                 num_of_samples):
        v_start = v[self.vehicle_key][i_start_coop]
        a_start = a[self.vehicle_key][i_start_coop]
        v_max = vehicle_list.veh_list[self.vehicle_key].v_max
        s_start = s[self.vehicle_key][i_start_coop]
        path = vehicle_list.veh_list[self.vehicle_key].path
        planing_horizon = float(i_end_coop - i_start_coop)*dt
        trajectories = self.generate_random_trajectories(planing_horizon, dt, num_of_samples, v_start, a_start, v_max,
                                                         s_start, path)

        i_best = self.select_best_trajectory(v, a, s, d, trajectories, i_start_coop, i_end_coop, dt, vehicle_list,
                                             relevant_veh)

        return trajectories.v[i_best], trajectories.a[i_best], trajectories.s_ref[i_best], trajectories.d[i_best]

    def generate_random_trajectories(self, planing_horizon, dt, number_of_samples, v_start, a_start, v_max, s_start, path):
        at_matrix, v_matrix = pvd_planner_util.random_vp(number_of_samples, planing_horizon, dt, v_start, a_start,
                                                         v_max)
        s_ref_matrix, d_matrix = self.calc_arc_coor_matrix(at_matrix, v_matrix, s_start, planing_horizon, dt, path)

        tra_samples = TrajectorySamples
        tra_samples.v = v_matrix
        tra_samples.a = at_matrix
        tra_samples.s_ref = s_ref_matrix
        tra_samples.d = d_matrix

        return tra_samples

    def calc_arc_coor_matrix(self, at_matrix, v_matrix, s_start, planing_horizon, dt, path):
        at_matrix_trans = np.transpose(at_matrix)
        v_matrix_trans = np.transpose(v_matrix)
        s_ref_matrix_trans = np.empty_like(at_matrix_trans)
        d_matrix_trans = np.empty_like(at_matrix_trans)
        s_ref_matrix_trans[0, :] = s_start
        d_matrix_trans[0, :] = path.get_d(s_start)

        for i in range(np.shape(s_ref_matrix_trans)[0] - 1):
            s, d = path.calc_arc_coord_next(v_matrix_trans[i, :], at_matrix_trans[i, :], dt, s_ref_matrix_trans[i, :])
            s_ref_matrix_trans[i + 1, :] = s
            d_matrix_trans[i + 1, :] = d

        s_ref_matrix = np.transpose(s_ref_matrix_trans)
        d_matrix = np.transpose(d_matrix_trans)

        return s_ref_matrix, d_matrix

    def select_best_trajectory(self, v, a, s, d, trajectory, i_start_coop, i_end_coop, dt, vehicle_list, relevant_veh):
        cost = np.zeros(np.shape(trajectory.v)[0])
        v_cost, a_cost = self.calc_dynamic_cost(trajectory, vehicle_list.veh_list[self.vehicle_key].cost_func, dt)
        sd_cost = self.calc_sd_costs(v, a, s, d, trajectory, i_start_coop, i_end_coop, vehicle_list, relevant_veh)

        cost += v_cost + a_cost + sd_cost
        i_best_trajectory = np.argmin(cost)
        #post_pro.unit_test_find_best_cooperation(trajectory.s_ref, trajectory.v, s[0], i_start_coop, i_end_coop, i_best_trajectory, i_end_coop-5, 1)
        return i_best_trajectory

    def calc_dynamic_cost(self, trajectories, cost_func, dt):
        v_cost_matrix = cost_func.v_comf_cost_pos.cost(trajectories.v) * dt + \
                        cost_func.v_comf_cost_neg.cost(trajectories.v) * dt + \
                        cost_func.v_discomf_cost_pos.cost(trajectories.v) * dt + \
                        cost_func.v_inf_cost_neg.cost(trajectories.v) * dt

        v_cost_array = np.sum(v_cost_matrix, axis=1)

        a_cost_matrix = cost_func.at_comf_cost_pos.cost(trajectories.a) * dt + \
                        cost_func.at_comf_cost_neg.cost(trajectories.a) * dt + \
                        cost_func.at_discomf_cost_pos.cost(trajectories.a) * dt + \
                        cost_func.at_discomf_cost_neg.cost(trajectories.a) * dt + \
                        cost_func.at_inf_cost_pos.cost(trajectories.a) * dt + \
                        cost_func.at_inf_cost_neg.cost(trajectories.a) * dt

        a_cost_array = np.sum(a_cost_matrix, axis=1)

        return v_cost_array, a_cost_array

    def calc_sd_costs(self, v, a, s, d, trajectory, i_start_coop, i_end_coop, vehicle_list, relevant_veh):
        # Calc samples sd-costs in relation with all veh in the list
        sd_cost = np.zeros(np.shape(trajectory.v)[0])
        for i in relevant_veh:
            samples_sd_cost = sd.sd_cost_samples_to_veh(trajectory.s_ref, s[i][i_start_coop:i_end_coop], trajectory.d,
                                                        d[i][i_start_coop:i_end_coop], trajectory.v,
                                                        vehicle_list.veh_list[self.vehicle_key], vehicle_list.veh_list[i],
                                                        vehicle_list.veh_list[self.vehicle_key].cost_func)
            sd_cost += samples_sd_cost

        # Calc ego sd-cost in relation with the samples
        ego_sd_cost = sd.sd_costs_veh_to_ego(s[0][i_start_coop : i_end_coop], trajectory.s_ref,
                                             d[0][i_start_coop : i_end_coop], trajectory.d, v[0][i_start_coop:i_end_coop],
                                             vehicle_list.veh_list[0], vehicle_list.veh_list[self.vehicle_key],
                                             vehicle_list.veh_list[0].cost_func)
        sd_cost += ego_sd_cost

        return sd_cost

    def idm_prediction(self, vehicle_list, i_start_idm, v, s, a, dt):
        # Determine preceding vehicle
        s_start = s[self.vehicle_key][i_start_idm]
        v_start = v[self.vehicle_key][i_start_idm]
        a_start = a[self.vehicle_key][i_start_idm]
        d_start = vehicle_list.veh_list[self.vehicle_key].path.get_d(s_start)
        path = vehicle_list.veh_list[self.vehicle_key].path
        v_opt = vehicle_list.veh_list[self.vehicle_key].v_opt
        idm_planing_horizon = float(np.shape(v)[1] - i_start_idm) * dt
        veh_length_to_fb = vehicle_list.veh_list[self.vehicle_key].properties.length_to_front_bumper

        prec_veh_key = None
        if self.vehicle_key > 1:
            prec_veh_key = self.vehicle_key -1
            if s[0][i_start_idm] - s_start > 0 and s[0][i_start_idm] - s_start < s[prec_veh_key][i_start_idm] - s_start:
                prec_veh_key = 0

        if prec_veh_key is not None:
            s_idm, v_idm, a_idm, d_idm = prediction.idm(s_start, d_start, v_start, a_start, v_opt, veh_length_to_fb,
                                                        s[prec_veh_key][i_start_idm:], v[prec_veh_key][i_start_idm:],
                                                        vehicle_list.veh_list[prec_veh_key].properties.length_to_rear_bumper,
                                                        dt, idm_planing_horizon, path)

        else:
            s_idm, v_idm, a_idm, d_idm = prediction.idm_free_drive(s_start, d_start, v_start, v_opt, dt, idm_planing_horizon, path)

        return v_idm, a_idm, s_idm, d_idm


class InnerOptimization(object):
    def __init__(self, veh_key, vehicle_list, relevant_veh, trajectorie_set, lc_info, sim_para):
        veh_start_state = trajectorie_set.get_veh_state(veh_key, lc_info.i_start_coop)
        random_trajectories = self.generate_random_trajectories(veh_start_state, vehicle_list[veh_key], lc_info, sim_para)

        costs = self.calc_cost_for_random_trajectories(vehicle_list[veh_key], vehicle_list, random_trajectories,
                                                       trajectorie_set, relevant_veh, sim_para, lc_info)
        i_best = np.argmin(costs)

        # post_pro.unit_test_find_best_cooperation(random_trajectories.s_ref, random_trajectories.v, trajectorie_set.s_ref[0],lc_info.i_start_coop, lc_info.i_end_coop, i_best, lc_info.i_lc, 1)

        self.solution = random_trajectories.get_trajectory(i_best)

    def generate_random_trajectories(self, veh_start_state, veh, lc_info, sim_para):
        planning_horizon_i_o = lc_info.t_end_coop + sim_para.sampling_range_t - lc_info.t_start_coop
        random_vps = pvd_planner_util.CreateVelocityProfiles(sim_para.num_of_samples, planning_horizon_i_o,
                                                             sim_para.sampling_range_t)
        random_vps.random_vp_by_jerk_given_start_state(veh_start_state.v, veh_start_state.a, veh.v_max)

        at_matrix_trans = np.transpose(random_vps.a_matrix)
        v_matrix_trans = np.transpose(random_vps.v_matrix)
        s_ref_matrix_trans = np.empty_like(at_matrix_trans)
        d_matrix_trans = np.empty_like(at_matrix_trans)
        s_ref_matrix_trans[0, :] = veh_start_state.s_ref
        d_matrix_trans[0, :] = veh.path.get_d(veh_start_state.s_ref)

        for i in range(np.shape(s_ref_matrix_trans)[0] - 1):
            s, d = veh.path.calc_arc_coord_next(v_matrix_trans[i, :], at_matrix_trans[i, :], sim_para.sampling_range_t,
                                                s_ref_matrix_trans[i, :])
            s_ref_matrix_trans[i + 1, :] = s
            d_matrix_trans[i + 1, :] = d

        s_ref_matrix = np.transpose(s_ref_matrix_trans)
        d_matrix = np.transpose(d_matrix_trans)

        return pvd_planner_util.TrajectorySamples(random_vps.a_matrix, random_vps.v_matrix, s_ref_matrix, d_matrix)

    def calc_cost_for_random_trajectories(self, veh, veh_list, random_trajectories, trajectorie_set, relevant_veh,
                                          sim_para, lc_info):
        dt = sim_para.sampling_range_t
        v_cost_matrix = veh.cost_func.v_comf_cost_pos.cost(random_trajectories.v) * dt + \
                        veh.cost_func.v_comf_cost_neg.cost(random_trajectories.v) * dt + \
                        veh.cost_func.v_discomf_cost_pos.cost(random_trajectories.v) * dt + \
                        veh.cost_func.v_inf_cost_neg.cost(random_trajectories.v) * dt

        v_cost_array = np.sum(v_cost_matrix, axis=1)

        a_cost_matrix = veh.cost_func.at_comf_cost_pos.cost(random_trajectories.a) * dt + \
                        veh.cost_func.at_comf_cost_neg.cost(random_trajectories.a) * dt + \
                        veh.cost_func.at_discomf_cost_pos.cost(random_trajectories.a) * dt + \
                        veh.cost_func.at_discomf_cost_neg.cost(random_trajectories.a) * dt + \
                        veh.cost_func.at_inf_cost_pos.cost(random_trajectories.a) * dt + \
                        veh.cost_func.at_inf_cost_neg.cost(random_trajectories.a) * dt

        a_cost_array = np.sum(a_cost_matrix, axis=1)

        # Calc samples sd-costs in relation with all veh in the list
        sd_cost = np.zeros(sim_para.num_of_samples, dtype=float)
        for i in relevant_veh:
            samples_sd_cost = sd.sd_cost_samples_to_veh(random_trajectories.s_ref,
                                                        trajectorie_set.s_ref[i, lc_info.i_start_coop:lc_info.i_end_coop+1],
                                                        random_trajectories.d,
                                                        trajectorie_set.d[i][lc_info.i_start_coop:lc_info.i_end_coop+1],
                                                        random_trajectories.v, veh, veh_list[i], veh.cost_func)
            sd_cost += samples_sd_cost

        # Calc ego sd-cost in relation with the samples
        ego_sd_cost = sd.sd_costs_veh_to_ego(trajectorie_set.s_ref[0][lc_info.i_start_coop: lc_info.i_end_coop+1],
                                             random_trajectories.s_ref,
                                             trajectorie_set.d[0][lc_info.i_start_coop: lc_info.i_end_coop+1],
                                             random_trajectories.d,
                                             trajectorie_set.v[0][lc_info.i_start_coop:lc_info.i_end_coop+1],
                                             veh_list[0], veh, veh_list[0].cost_func)
        sd_cost += ego_sd_cost

        cost_array = v_cost_array + a_cost_array + sd_cost

        return cost_array
