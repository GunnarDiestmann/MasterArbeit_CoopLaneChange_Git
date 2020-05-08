import numpy as np
import path_planing
import coop_prediction
import prediction
import safety_distance
import pvd_planner_util
import veh_classification
import post_pro
import matplotlib.pyplot as plt


class LaneChangeInfo(object):
    def __init__(self, t_lc, t_start_lc, t_start_coop, t_end_coop, i_lc, i_start_coop, i_end_coop, i_start_lc,
                 involved_veh):
        self.t_lc = t_lc
        self.t_start_lc = t_start_lc
        self.t_start_coop = t_start_coop
        self.t_end_coop = t_end_coop

        self.i_lc = i_lc
        self.i_start_lc = i_start_lc
        self.i_start_coop = i_start_coop
        self.i_end_coop = i_end_coop

        self.involved_veh = involved_veh


class CooperativeSolution(object):
    def __init__(self, veh_prediction, ego_vp, len_vehicle_list, sim_para):
        rows = len_vehicle_list
        columns = sim_para.num_of_time_steps

        v = np.empty((rows, columns), dtype=float)
        a = np.empty_like(v)
        s_ref = np.empty_like(v)
        d = np.empty_like(v)

        v[0, :] = ego_vp.v
        a[0, :] = ego_vp.a

        for i in range(1, len_vehicle_list):
            v[i] = veh_prediction.v[i - 1]
            a[i] = veh_prediction.a[i - 1]
            s_ref[i] = veh_prediction.s_ref[i - 1]
            d[i] = veh_prediction.d[i - 1]

        self.trajectorie_set = pvd_planner_util.TrajectorySet(v, a, s_ref, d, sim_para.sampling_range_t)

        self.sim_para = sim_para

        self.path_plan = None
        self.ego_path = None

        self.lc_initiated = False
        self.s_start_change = None

        self.lc_info = None
        self.cost = 0

    def acceleration_lane_planer(self, street_info, vehicle_list, ego_vp):
        # Path planing
        self.plan_lane_change_path(ego_vp, vehicle_list.veh_list[0], street_info)

        # post_pro.plot_trajectory_set(self.trajectorie_set, self.sim_para.sampling_range_t)

        # Classification
        classification = veh_classification.Classification()
        classification.classify_for_lane_change(self.trajectorie_set.s_ref, self.trajectorie_set.v, vehicle_list, self.path_plan.i_lane_change)

        self.create_lane_change_info(classification)

        # Cooperative prediction
        self.cooperative_trajectory_for_involved_veh(classification, vehicle_list.veh_list)

        # post_pro.plot_trajectory_set(self.trajectorie_set, self.sim_para.sampling_range_t)

        # Reprediction after lane change
        if classification.involved_veh_keys:
            self.idm_prediction_involved(classification, vehicle_list.veh_list)
        if classification.influenced_veh_keys:
            self.idm_prediction_influenced(classification.influenced_veh_keys, self.lc_info.i_lc, self.lc_info.i_start_coop,
                                           self.sim_para.num_of_time_steps, vehicle_list.veh_list, self.sim_para.sampling_range_t)

        # post_pro.plot_trajectory_set(self.trajectorie_set, self.sim_para.sampling_range_t)

        # Calc cost
        self.calc_cost_for_solution(vehicle_list, classification.involved_veh_keys)

    def plan_lane_change_path(self, ego_vp, ego_veh, street_info):
        self.path_plan = path_planing.PathPlaner(self.sim_para.num_of_time_steps)
        s_change_min_replan = ego_veh.s_start
        for i in range(self.sim_para.time_steps_in_planing_step):
            s_next, d_next = ego_veh.path.calc_arc_coord_next(ego_vp.v[i], ego_vp.a[i], self.sim_para.sampling_range_t,
                                                              s_change_min_replan)
            s_change_min_replan = s_next

        self.s_change_min_replan = s_change_min_replan

        self.path_plan.plane_acceleration_lane_path(street_info, ego_vp, ego_veh.s_start, self.sim_para.sampling_range_t,
                                               s_change_min_replan, ego_veh.properties.width)

        self.s_start_change = self.path_plan.s_start_change
        self.trajectorie_set.s_ref[0, :] = self.path_plan.s_ref
        self.trajectorie_set.d[0, :] = self.path_plan.d
        self.ego_path = self.path_plan.path

    def create_lane_change_info(self, classification):
        i_interact = self.path_plan.i_lane_change
        i_start_coop = max(0, i_interact - int(self.sim_para.t_start_coop_before_ls / self.sim_para.sampling_range_t))
        i_end_coop = self.path_plan.i_end_change

        self.i_interact = i_interact
        self.lc_initiated = i_interact < 2 * self.sim_para.time_steps_in_planing_step + 1

        t_lc = i_interact * self.sim_para.sampling_range_t
        t_start_lc = self.path_plan.i_start_change * self.sim_para.sampling_range_t
        t_start_coop = i_start_coop * self.sim_para.sampling_range_t
        t_end_coop = i_end_coop * self.sim_para.sampling_range_t
        self.lc_info = LaneChangeInfo(t_lc, t_start_lc, t_start_coop, t_end_coop, i_interact, i_start_coop, i_end_coop,
                                      self.path_plan.i_start_change, classification.involved_veh_keys)

    def cooperative_trajectory_for_involved_veh(self, classification, vehicle_list):
        take_into_account = classification.unaffected_veh_key + [0]
        for i in classification.involved_veh_keys:
            inner_optimization = coop_prediction.InnerOptimization(i, vehicle_list, take_into_account,
                                                                   self.trajectorie_set, self.lc_info, self.sim_para)
            coop_trajectory = inner_optimization.solution

            self.trajectorie_set.change_trajectory_from_to(i, self.lc_info.i_start_coop, self.lc_info.i_end_coop+1,
                                                           coop_trajectory)
            take_into_account += [i]

    def idm_prediction_involved(self, classification, veh_list):

        for i in classification.involved_veh_keys:
            veh_state = self.trajectorie_set.get_veh_state(i, self.lc_info.i_end_coop)
            prec_veh_key = None
            if i > 1:
                prec_veh_key = i - 1
                if 0 < self.trajectorie_set.s_ref[0][self.lc_info.i_end_coop] - veh_state.s_ref < \
                        self.trajectorie_set.s_ref[prec_veh_key][self.lc_info.i_end_coop] - veh_state.s_ref:
                    prec_veh_key = 0


            if prec_veh_key is not None:
                prec_veh = veh_list[prec_veh_key]
                prec_veh_trajectory = self.trajectorie_set.get_trajectory_from_to(prec_veh_key, self.lc_info.i_end_coop,
                                                                                  self.sim_para.num_of_time_steps+1)
                trajectory = prediction.idm(veh_list[i], veh_state, prec_veh, prec_veh_trajectory,
                                            self.sim_para.sampling_range_t)

            else:
                idm_planing_horizon = self.sim_para.planning_horizon + self.sim_para.sampling_range_t - self.lc_info.t_end_coop
                trajectory = prediction.idm_free_drive(veh_list[i], veh_state, self.sim_para.sampling_range_t,
                                                       idm_planing_horizon)
            self.trajectorie_set.change_trajectory_from_to(i, self.lc_info.i_end_coop, self.sim_para.num_of_time_steps+1,
                                                           trajectory)

    def idm_prediction_influenced(self, veh_keys, i_interact, i_start_coop, i_max, veh_list, dt):
        # First vehicle
        influ_veh_key = veh_keys[0]
        influ_veh = veh_list[influ_veh_key]

        # if ego is new preceding veh, idm starts when ego_veh enters the lane. so its i_interact
        if veh_keys[0] == 1 or self.trajectorie_set.s_ref[0][i_interact] < self.trajectorie_set.s_ref[veh_keys[0]-1][i_interact]:
            prec_veh = veh_list[0]
            prec_veh_trajectory = self.trajectorie_set.get_trajectory_from_to(0, i_interact, i_max)
            influ_veh_start_state = self.trajectorie_set.get_veh_state(influ_veh_key, i_interact)

            idm_trajectory = prediction.idm(influ_veh, influ_veh_start_state, prec_veh, prec_veh_trajectory, dt)

            self.trajectorie_set.change_trajectory_from_to(influ_veh_key, i_interact, i_max, idm_trajectory)

        # if other vehicle ist preceding veh than idm starts at i_start_coop
        else:
            prec_veh_key = veh_keys[0] - 1
            prec_veh = veh_list[prec_veh_key]
            prec_veh_trajectory = self.trajectorie_set.get_trajectory_from_to(prec_veh_key, i_start_coop, i_max)
            influ_veh_start_state = self.trajectorie_set.get_veh_state(influ_veh_key, i_start_coop)

            idm_trajectory = prediction.idm(influ_veh, influ_veh_start_state, prec_veh, prec_veh_trajectory, dt)

            self.trajectorie_set.change_trajectory_from_to(influ_veh_key, i_start_coop, i_max, idm_trajectory)

        # following vehicle
        for i in range(1, len(veh_keys)):
            influ_veh_key = veh_keys[i]
            influ_veh = veh_list.veh_list[influ_veh_key]
            influ_veh_start_state = self.trajectorie_set.get_veh_state(influ_veh_key, i_start_coop)

            prec_veh_key = veh_keys[i] - 1
            prec_veh = veh_list[prec_veh_key]
            prec_veh_trajectory = self.trajectorie_set.get_trajectory_from_to(prec_veh_key, i_start_coop, i_max)

            idm_trajectory = prediction.idm(influ_veh, influ_veh_start_state, prec_veh, prec_veh_trajectory,dt)

            self.trajectorie_set.change_trajectory_from_to(influ_veh_key, i_start_coop, i_max, idm_trajectory)

    def calc_cost_for_solution(self, vehicle_list, veh_keys_involved):
        veh_list = vehicle_list.veh_list
        cost = 0
        # Calculate cost for ego veh
        # Dynamic costs
        ego_trajectory = self.trajectorie_set.get_veh_trajectory(0)
        cost += veh_list[0].cost_func.calc_at_costs(ego_trajectory.a)
        cost += veh_list[0].cost_func.calc_v_costs(ego_trajectory.v)
        # Interactive costs
        # Costs safety distance costs
        for i in range(1, len(veh_list)):
            other_veh_trajectory = self.trajectorie_set.get_veh_trajectory(i)
            cost += safety_distance.sd_cost_veh_to_veh(ego_trajectory.s_ref, other_veh_trajectory.s_ref,
                                                       ego_trajectory.d, other_veh_trajectory.d,
                                                       ego_trajectory.v, veh_list[0], veh_list[i],veh_list[0].cost_func)
        # Lc Costs
        follow_veh_target_lane = None
        for i in range(1, vehicle_list.num_of_veh_tl+1):
            if self.trajectorie_set.s_ref[i, self.lc_info.i_lc] < self.trajectorie_set.s_ref[0, self.lc_info.i_lc]:
                follow_veh_target_lane = i
                break
        if follow_veh_target_lane is not None:
            ego_veh_state_lc = self.trajectorie_set.get_veh_state(0, self.lc_info.i_lc)
            follow_veh_state_lc = self.trajectorie_set.get_veh_state(follow_veh_target_lane, self.lc_info.i_lc)
            t_to_lc_end = self.lc_info.t_end_coop - self.lc_info.t_lc
            cost += veh_list[0].cost_func.calc_lc_costs(ego_veh_state_lc, follow_veh_state_lc, t_to_lc_end)

        # Calculate costs for cooperative vehicle
        for i in veh_keys_involved:
            # Dynamic costs
            veh_trajectory = self.trajectorie_set.get_veh_trajectory(i)
            cost += veh_list[i].cost_func.calc_at_costs(veh_trajectory.a)
            cost += veh_list[i].cost_func.calc_v_costs(veh_trajectory.v)
            j_list = range(i) + range(i+1, len(veh_list))
            # Interactive costs
            for j in j_list:
                other_veh_trajectory = self.trajectorie_set.get_veh_trajectory(j)
                cost += safety_distance.sd_cost_veh_to_veh(veh_trajectory.s_ref, other_veh_trajectory.s_ref,
                                                           veh_trajectory.d, other_veh_trajectory.d,
                                                           veh_trajectory.v, veh_list[i], veh_list[j],
                                                           veh_list[i].cost_func)

        self.cost += cost
