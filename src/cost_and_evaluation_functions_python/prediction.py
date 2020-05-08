import numpy as np
import pvd_planner_util


def idm_free_drive(veh, veh_start_state, dt, planing_horizon, acc_comf=2., delta=4.):
    planing_steps = int(round(planing_horizon/dt))

    s_veh = np.zeros(planing_steps, dtype=float)
    s_veh[0] = veh_start_state.s_ref
    a_veh = np.zeros_like(s_veh, dtype=float)
    a_veh[0] = veh_start_state.a
    v_veh = np.zeros_like(s_veh, dtype=float)
    v_veh[0] = veh_start_state.v
    d_veh = np.zeros_like(s_veh, dtype=float)
    d_veh[0] = veh_start_state.d

    for i in range(1, planing_steps):
        v_veh[i] = v_veh[i-1] + a_veh[i-1] * dt
        s_veh[i], d_veh[i] = veh.path.calc_arc_coord_next(v_veh[i-1], a_veh[i-1], dt, s_veh[i-1])
        a_veh[i] = acc_comf * (1 - (v_veh[i] / veh.v_opt) ** delta)

    return pvd_planner_util.Trajectory(a_veh, v_veh, s_veh, d_veh)


def idm(veh, veh_start_state, prec_veh, prec_veh_trajectory, dt, acc_comf=2., break_comf=2., delta=4., a_min=-10):
    planing_steps = len(prec_veh_trajectory.v)
    s_rear_bump = prec_veh_trajectory.s_ref - prec_veh.properties.length_to_rear_bumper

    s_veh = np.zeros(planing_steps, dtype=float)
    s_veh[0] = veh_start_state.s_ref

    a_veh = np.zeros_like(s_veh, dtype=float)
    a_veh[0] = veh_start_state.a

    v_veh = np.zeros_like(s_veh, dtype=float)
    v_veh[0] = veh_start_state.v

    d_veh = np.zeros_like(s_veh, dtype=float)
    d_veh[0] = veh_start_state.d

    for i in range(1, planing_steps):
        v_veh[i] = max(v_veh[i-1] + a_veh[i-1] * dt, 0)
        s_veh[i], d_veh[i] = veh.path.calc_arc_coord_next(v_veh[i-1], a_veh[i-1], dt, s_veh[i-1])

        min_distance = 5
        sd = max(v_veh[i] * 3.6 * 0.5 - min_distance, 0.)
        s_net = s_rear_bump[i] - (s_veh[i] + veh.properties.length_to_front_bumper)
        delta_v = v_veh[i] - prec_veh_trajectory.v[i]
        s_star = sd + min_distance + (v_veh[i] * delta_v) / (2 * (acc_comf * break_comf) ** 0.5)
        # Calc a according to IDM but interaction term just if distance is smaller than wanted distance.
        # Negative acceleration is restricted to a_min
        a_veh[i] = max(acc_comf * (1 - (v_veh[i] / veh.v_opt) ** delta - ((s_star / s_net) * (s_star > s_net)) ** 2), a_min)
        # Restrict a so that no negative velocity will be generated
        a_veh[i] += - (a_veh[i] + (v_veh[i]/dt)) * (v_veh[i] + a_veh[i] * dt < 0)
        # Restrict a so that no jerk higher than 2 will be created
        a_dif = a_veh[i] - a_veh[i-1]
        a_veh[i] += (-a_veh[i] + a_veh[i-1] + 3 * np.sign(a_dif) * dt) * (abs(a_dif)/dt > 3)
    return pvd_planner_util.Trajectory(a_veh, v_veh, s_veh, d_veh)


class LaneBasedPrediction(object):
    def __init__(self, num_of_veh, sim_para):
        rows = int(num_of_veh)
        columns = sim_para.num_of_time_steps
        self.v = np.empty([rows, columns], dtype=float)
        self.a = np.empty_like(self.v)
        self.s_ref = np.empty_like(self.v)
        self.d = np.empty_like(self.v)

        self.sim_para = sim_para

    def idm_prediction(self, vehicle_list):
        for i in range(1, len(vehicle_list.veh_list)):
            self.v[i-1, 0] = vehicle_list.veh_list[i].v_start
            self.a[i - 1, 0] = vehicle_list.veh_list[i].a_start
            self.s_ref[i - 1, 0] = vehicle_list.veh_list[i].s_start
            self.d[i - 1, 0] = vehicle_list.veh_list[i].d_start

        # predict vehicle on target lane
        if vehicle_list.num_of_veh_tl != 0:
            self.idm_prediction_lane(vehicle_list.veh_list[1:1+vehicle_list.num_of_veh_tl], 0)

        # predict vehicle on initial lane
        if vehicle_list.num_of_veh_il != 0:
            j_start = vehicle_list.num_of_veh_tl + 1
            self.idm_prediction_lane(vehicle_list[j_start:1 + j_start + vehicle_list.num_of_veh_il], j_start-1)

    def idm_prediction_lane(self, veh_list, j):
        # Predict fist vehicle
        veh_start_state = pvd_planner_util.VehicleState(veh_list[j].a_start, veh_list[j].v_start, veh_list[j].s_start,
                                                        veh_list[j].d_start)
        idm_trajectory = idm_free_drive(veh_list[j], veh_start_state, self.sim_para.sampling_range_t,
                                        self.sim_para.planning_horizon)

        self.v[j, :] = idm_trajectory.v
        self.s_ref[j, :] = idm_trajectory.s_ref
        self.a[j, :] = idm_trajectory.a
        self.d[j, :] = idm_trajectory.d

        # Predict following vehicle
        for i in range(1, len(veh_list)):
            veh_start_state = pvd_planner_util.VehicleState(veh_list[j+i].a_start, veh_list[j+i].v_start,
                                                            veh_list[j+i].s_start, veh_list[j+i].d_start)
            prec_veh_trajectory = pvd_planner_util.Trajectory(self.a[j+i-1], self.v[j+i-1], self.s_ref[j+i-1],
                                                              self.d[j+i-1])

            idm_trajectory = idm(veh_list[j+i], veh_start_state, veh_list[j+i-1], prec_veh_trajectory,
                                 self.sim_para.sampling_range_t)
            self.v[j+i, :] = idm_trajectory.v
            self.s_ref[j+i, :] = idm_trajectory.s_ref
            self.a[j+i, :] = idm_trajectory.a
            self.d[j+i, :] = idm_trajectory.d
