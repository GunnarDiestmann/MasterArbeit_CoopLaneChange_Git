import pvd_planner_util
import prediction
import numpy as np
import scenario_acceleration_lane2
import matplotlib.pyplot as plt


def safety_distance(v_r, v_f, t_resp=1., a_max_acc=7., a_min_break=9, a_max_brake=10):
    sd = v_r * t_resp + 0.5 * a_max_acc * t_resp ** 2 + (v_r + t_resp * a_max_acc)**2/(2 * a_min_break) - v_f **2 / \
         2 * a_max_brake
    return sd


class VehicleState(object):
    def __init__(self, s_ref, v, a, path, vehicle_properties):
        self.s_ref = s_ref
        self.v = v
        self.a = a
        self.path = path
        self.vehicle_properties = vehicle_properties


class SafetyCheckLC(object):
    def __init__(self, ego_state, s_lane_end, ego_state_lc, prec_state, following_state, delta_t_lc,  dt = 0.1):
        self.ego_state = ego_state
        self.s_lane_end = s_lane_end
        self.ego_state_lc = ego_state_lc
        self.prec_state = prec_state
        self.following_state = following_state
        self.delta_t_lc = delta_t_lc
        self.dt = dt

    def planB_check(self, b_krit=2., s_stop_before_lane_end=0.):
        s_stop = self.s_lane_end - s_stop_before_lane_end

        s_break = s_stop - self.ego_state.s_ref

        mean_deceleration = (self.ego_state.s_ref.v ** 2) / (2 * s_break)

        return mean_deceleration < b_krit

    def safety_check(self):
        safety_check_prec = self.safety_check_prec()
        safety_check_following = self.safety_check_following()

        return safety_check_prec and safety_check_following

    def safety_check_prec(self):
        s_prec_lc = self.prec_state.s_ref
        v = self.prec_state.v
        if self.prec_state.a < 0:
            a = self.prec_state.a
        else:
            a = 0
        for i in range(int(round(self.delta_t_lc/self.dt))):
            s_prec_lc, d = self.prec_state.path.calc_arc_coord_next(v, a, self.dt, s_prec_lc)
            v = v+a*self.dt

        d_min_rss = safety_distance(self.ego_state_lc.v, v)
        distance = (s_prec_lc - self.prec_state.vehicle_properties.length_to_rear_bumper) - (
                    self.ego_state_lc.s_ref + self.ego_state.length_to_front_bumper)

        return distance > d_min_rss

    def safety_check_following(self):
        s_following_lc = self.following_state.s_ref
        v = self.following_state.v
        if self.following_state.a < 0:
            a = self.following_state.a
        else:
            a = 0
        for i in range(int(round(self.delta_t_lc / self.dt))):
            s_following_lc, d = self.following_state.path.calc_arc_coord_next(v, a, self.dt, s_following_lc)
            v = v + a * self.dt

        d_min_rss = safety_distance(v, self.ego_state_lc.v)
        distance = (self.ego_state_lc.s_ref - self.ego_state.vehicle_properties.length_to_rear_bumper) - (
                    s_following_lc + self.following_state.length_to_front_bumper)

        return distance > d_min_rss


class TrajectorySetReplan(object):
    def __init__(self, vehicle_list, sampling_range_t):
        self.vehicle_id_list = []
        self.v = []
        self.a = []
        self.s_ref = []
        self.d = []
        self.paths = []
        self.ego_paths_at_time_stemps = []
        self.sampling_range_t = sampling_range_t

        for i in range(len(vehicle_list)):
            self.vehicle_id_list.append(vehicle_list[i].veh_index)
            self.v.append([])
            self.a.append([])
            self.s_ref.append([])
            self.d.append([])
            self.paths.append([])

    def add_to_trajectories(self, v, a, s_ref, d):
        for i in range(len(v)):
            self.v[i].extend(v[i])
            self.a[i].extend(a[i])
            self.s_ref[i].extend(s_ref[i])
            self.d[i].extend(d[i])

    def add_to_trajectory(self, veh_id, v, a, s_ref, d):
        list_index = self.determine_list_index_to_veh_id(veh_id)
        self.v[list_index].extend(v)
        self.a[list_index].extend(a)
        self.s_ref[list_index].extend(s_ref)
        self.d[list_index].extend(d)

    def set_paths(self, paths):
        for i in range(len(paths)):
            self.paths[i] = paths[i]

    def set_path(self, veh_id, path):
        list_index = self.determine_list_index_to_veh_id(veh_id)
        self.paths[list_index] = path

    def get_states_at_time(self, t):
        v = []
        a = []
        s_ref = []
        d = []
        t_index = self.determine_list_index_to_time(t)
        for i in range(len(self.vehicle_id_list)):
            v.append(self.v[i][t_index])
            a.append(self.a[i][t_index])
            s_ref.append(self.s_ref[i][t_index])
            d.append(self.d[i][t_index])
        return self.vehicle_id_list, v, a, s_ref, d

    def get_states_from_to(self, t_start, t_duration):
        t_index_start = self.determine_list_index_to_time(t_start)
        t_index_end = self.determine_list_index_to_time(t_start + t_duration)
        v = []
        a = []
        s_ref = []
        d = []
        for i in range(len(self.vehicle_id_list)):
            v.append(self.v[i][t_index_start:t_index_end])
            a.append(self.a[i][t_index_start:t_index_end])
            s_ref.append(self.s_ref[i][t_index_start:t_index_end])
            d.append(self.d[i][t_index_start:t_index_end])
        return self.vehicle_id_list, v, a, s_ref, d

    def get_state_of_veh_from_to(self, veh_id, t_start, t_duration):
        t_index_start = self.determine_list_index_to_time(t_start)
        t_index_end = self.determine_list_index_to_time(t_start + t_duration)
        veh_list_index = self.determine_list_index_to_veh_id(veh_id)

        v = self.v[veh_list_index][t_index_start:t_index_end]
        a = self.a[veh_list_index][t_index_start:t_index_end]
        s_ref = self.s_ref[veh_list_index][t_index_start:t_index_end]
        d = self.d[veh_list_index][t_index_start:t_index_end]

        return v, a, s_ref, d

    def determine_list_index_to_veh_id(self, veh_id):
        for i in range(len(self.vehicle_id_list)):
            if veh_id == self.vehicle_id_list[i]:
                return i

    def determine_list_index_to_time(self, t):
        return int(round(t/self.sampling_range_t))


class Replanning(object):
    def __init__(self, sim_para, vehicle_list, ego_start_vp, street_info):
        self.sim_para = sim_para
        self.vehicle_list = vehicle_list
        self.trajectory_set = TrajectorySetReplan(vehicle_list.veh_list, sim_para.sampling_range_t)
        self.street_info = street_info
        self.ego_start_vp = ego_start_vp
        self.replan_type = None
        self.t = 0

    def multi_agent_optimum_replan(self):
        self.replan_type = 0
        for i in range(len(self.vehicle_list.veh_list)):
            veh = self.vehicle_list.veh_list[i]
            self.trajectory_set.add_to_trajectory(veh.veh_index, [veh.v_start], [veh.a_start], [veh.s_start], [veh.d_start])

        paths = []
        for i in range(len(self.vehicle_list.veh_list)):
            paths.append(self.vehicle_list.veh_list[i].path)
        self.trajectory_set.set_paths(paths)

        self.start_replan()

    def egoistic_replan(self):
        # egoistic prediction
        sref_prediction = []
        d_prediction = []
        v_prediction = []
        a_prediction = []
        # leading vehicle
        lead_veh = self.vehicle_list.veh_list[1]
        s_idm, v_idm, a_idm, d_idm = prediction.idm_free_drive(lead_veh.s_start, lead_veh.d_start, lead_veh.v_start,
                                                               lead_veh.v_opt, self.sampling_range_t,
                                                               self.planning_horizon + 10.,
                                                               lead_veh.path)
        sref_prediction.append(s_idm)
        d_prediction.append(d_idm)
        v_prediction.append(v_idm)
        a_prediction.append(a_idm)

        for i in range(2, self.vehicle_list.num_of_veh_tl + 1):
            veh = self.vehicle_list.veh_list[i]
            prec_veh = self.vehicle_list.veh_list[i - 1]
            s_idm, v_idm, a_idm, d_idm, = prediction.idm(veh.s_start, veh.d_start,
                                                         veh.properties.length_to_front_bumper, sref_prediction[-1],
                                                         prec_veh.properties.length_to_rear_bumper, veh.v_start,
                                                         v_prediction[-1], veh.v_opt,
                                                         self.sampling_range_t, self.planning_horizon + 15., veh.path)
            sref_prediction.append(s_idm)
            d_prediction.append(d_idm)
            v_prediction.append(v_idm)
            a_prediction.append(a_idm)

        for i in range(1, len(self.vehicle_list.veh_list)):
            self.trajectory_set.add_to_trajectory(self.vehicle_list.veh_list[i].veh_index, v_prediction[i - 1],
                                                  a_prediction[i - 1], sref_prediction[i - 1], d_prediction[i - 1])

        self.trajectory_set.add_to_trajectory(0, self.ego_v_start_list, self.ego_a_start_list,
                                              self.ego_s_ref_start_list,
                                              self.ego_d_start_list)

        paths = []
        for i in range(len(self.vehicle_list.veh_list)):
            paths.append(self.vehicle_list.veh_list[i].path)
        self.trajectory_set.set_paths(paths)

        self.start_replan()

    def start_replan(self):
        bool_lane_change_init = False

        while not bool_lane_change_init:
            coop_lane_change_solution = scenario_acceleration_lane2.cooperative_lane_change(self.vehicle_list,
                                        self.ego_start_vp, self.street_info, self.sim_para)

            plt.figure()

            ax_s = plt.subplot2grid((13, 1), (0, 0), rowspan=4)
            ax_d = plt.subplot2grid((13, 1), (5, 0), rowspan=2, sharex=ax_s)
            ax_v = plt.subplot2grid((13, 1), (8, 0), rowspan=2, sharex=ax_s)
            ax_a = plt.subplot2grid((13, 1), (11, 0), rowspan=2, sharex=ax_s)

            t_list = np.empty(np.size(coop_lane_change_solution.trajectorie_set.v, axis=1))
            for i in range(len(t_list)):
                t_list[i] = i*self.sim_para.sampling_range_t
            for i in range(np.size(coop_lane_change_solution.trajectorie_set.v, axis=0)):
                ax_s.plot(t_list, coop_lane_change_solution.trajectorie_set.s_ref[i, :])
                ax_d.plot(t_list, coop_lane_change_solution.trajectorie_set.d[i, :])
                ax_v.plot(t_list, coop_lane_change_solution.trajectorie_set.v[i, :])
                ax_a.plot(t_list, coop_lane_change_solution.trajectorie_set.a[i, :])

            plt.xlabel('t [s]')
            ax_s.set_ylabel('s [m]')
            ax_d.set_ylabel('d [m]')
            ax_v.set_ylabel('v [m/s]')
            ax_a.set_ylabel('a [m/s^2]')

            figname = '/home/diestmann/Documents/post_pro/Replan/' + str(int(self.t))
            plt.show()
            plt.savefig(figname)
            plt.close()

            if coop_lane_change_solution.lc_initiated:
                bool_lane_change_init = True
                if self.replan_type == 0:
                    self.set_values_until_end_mao(coop_lane_change_solution)
                elif self.replan_type == 1:
                    self.set_values_until_end_egoistic(coop_lane_change_solution)

            else:
                self.t += self.sim_para.t_replan
                print "time:", self.t
                print "time of planed lc:", coop_lane_change_solution.lc_info.t_lc
                print "s planed lc:", coop_lane_change_solution.s_start_change
                print "cost:", coop_lane_change_solution.cost
                print "t_start_coop: ", coop_lane_change_solution.lc_info.t_start_coop

                self.trajectory_set.ego_paths_at_time_stemps.append(coop_lane_change_solution.ego_path)
                self.vehicle_list.veh_list[0].path = coop_lane_change_solution.ego_path
                if self.replan_type == 0:
                    self.save_values_mao(coop_lane_change_solution)
                elif self.replan_type == 1:
                    self.save_values_egoistic(coop_lane_change_solution)

                self.set_new_starting_states(coop_lane_change_solution)

    def save_values_mao(self, coop_lane_change_solution):
        t_start_i = 1
        t_end_i = self.sim_para.time_steps_in_planing_step + t_start_i
        v = []
        a = []
        s_ref = []
        d = []

        for i in range(len(self.vehicle_list.veh_list)):
            v.append(coop_lane_change_solution.trajectorie_set.v[i, t_start_i:t_end_i])
            a.append(coop_lane_change_solution.trajectorie_set.a[i, t_start_i:t_end_i])
            s_ref.append(coop_lane_change_solution.trajectorie_set.s_ref[i, t_start_i:t_end_i])
            d.append(coop_lane_change_solution.trajectorie_set.d[i, t_start_i:t_end_i])

        self.trajectory_set.add_to_trajectories(v, a, s_ref, d)

    def save_values_egoistic(self, coop_lane_change_solution):
        t_start_i = 1
        t_end_i = self.sim_para.time_step_in_plannning_step + t_start_i

        v = coop_lane_change_solution.v[0, t_start_i:t_end_i]
        a = coop_lane_change_solution.a[0, t_start_i:t_end_i]
        s_ref = coop_lane_change_solution.s_ref[0, t_start_i:t_end_i]
        d = coop_lane_change_solution.d[0, t_start_i:t_end_i]

        self.trajectory_set.add_to_trajectory(0, v, a, s_ref, d)

    def set_values_until_end_mao(self, coop_lane_change_solution):
        t_start_i = 1
        v = []
        a = []
        s_ref = []
        d = []
        for i in range(len(self.vehicle_list.veh_list)):
            v.append(coop_lane_change_solution.v[i, t_start_i:])
            a.append(coop_lane_change_solution.a[i, t_start_i:])
            s_ref.append(coop_lane_change_solution.s_ref[i, t_start_i:])
            d.append(coop_lane_change_solution.d[i, t_start_i:])

        self.trajectory_set.add_to_trajectories(v, a, s_ref, d)

    def set_values_until_end_egoistic(self, coop_lane_change_solution):
        t_start_i = 1

        v = coop_lane_change_solution.v[0, t_start_i:]
        a = coop_lane_change_solution.a[0, t_start_i:]
        s_ref = coop_lane_change_solution.s_ref[0, t_start_i:]
        d = coop_lane_change_solution.d[0, t_start_i:]

        self.trajectory_set.add_to_trajectory(0, v, a, s_ref, d)

    def set_new_starting_states(self, coop_lane_change_solution):
        veh_id_list, v, a, s_ref, d = self.trajectory_set.get_states_at_time(self.t)
        for i in range(len(self.vehicle_list.veh_list)):
            self.vehicle_list.veh_list[i].a_start = a[i]
            self.vehicle_list.veh_list[i].v_start = v[i]
            self.vehicle_list.veh_list[i].s_start = s_ref[i]
            self.vehicle_list.veh_list[i].d_start = d[i]

        list_index_t_now = self.sim_para.time_step_in_plannning_step
        list_index_t_end = list_index_t_now + self.sim_para.time_step_in_plannning_step
        v_ego = coop_lane_change_solution.v[0, list_index_t_now:list_index_t_end]
        a_ego = coop_lane_change_solution.a[0, list_index_t_now:list_index_t_end]
        s_ref_ego = coop_lane_change_solution.s_ref[0, list_index_t_now:list_index_t_end]
        d_ego = coop_lane_change_solution.d[0, list_index_t_now:list_index_t_end]

        self.ego_d_start_list = d_ego
        self.ego_s_ref_start_list = s_ref_ego
        self.ego_v_start_list = v_ego
        self.ego_a_start_list = a_ego
