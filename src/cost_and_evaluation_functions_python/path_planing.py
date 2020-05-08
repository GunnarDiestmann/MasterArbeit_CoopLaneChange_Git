import numpy as np
import general_functions as gfunc
import Polynomial_funktion
import math
import post_pro


class Path(object):
    def __init__(self, path_func, street_info):
        self.path_func = path_func
        self.street_info = street_info

    def calc_arc_coord_next(self, v, a, dt, s_curr):
        path_dt = (self.path_func(s_curr + 0.1) - self.path_func(s_curr - 0.1)) / (2 * 0.1)
        theta_path = np.arctan(path_dt)

        cl_curvature = self.street_info.get_center_line_curvature(s_curr)

        s_ref_next = s_curr + (np.cos(theta_path) * gfunc.calc_ds(dt, v, a)) / (1 - cl_curvature * self.path_func(s_curr))
        d_next = self.path_func(s_ref_next)

        return s_ref_next, d_next

    def get_d(self, s):
        return self.path_func(s)

    def get_yaw(self, s_ref):
        cl_yaw = self.street_info.get_center_line_yaw(s_ref)
        path_delta_d = (self.path_func(s_ref + 0.1) - self.path_func(s_ref - 0.1)) / (2 * 0.1)
        path_yaw = np.arctan(path_delta_d)
        yaw = cl_yaw + path_yaw
        return yaw


class MultiPath(object):
    def __init__(self, init_path):
        self.init_path = init_path
        self.transition_path = None
        self.target_path = None
        self.s_init_to_trans = float("inf")
        self.s_trans_to_target = float("inf")

    def calc_arc_coord_next(self, v, a, dt, s_curr):
        if s_curr < self.s_init_to_trans:
            return self.init_path.calc_arc_coord_next(v, a, dt, s_curr)
        elif s_curr < self.s_trans_to_target:
            return self.transition_path.calc_arc_coord_next(v, a, dt, s_curr)
        else:
            return self.target_path.calc_arc_coord_next(v, a, dt, s_curr)

    def get_d(self, s):
        if s < self.s_init_to_trans:
            return self.init_path.get_d(s)
        elif s < self.s_trans_to_target:
            return self.transition_path.get_d(s)
        else:
            return self.target_path.get_d(s)

    def get_yaw(self, s_ref):
        if s_ref < self.s_init_to_trans:
            return self.init_path.get_yaw(s_ref)
        elif s_ref < self.s_trans_to_target:
            return self.transition_path.get_yaw(s_ref)
        else:
            return self.target_path.get_yaw(s_ref)


class PathPlaner(object):
    def __init__(self, num_of_time_steps):
        self.s_ref = np.empty(num_of_time_steps)
        self.d = np.empty(num_of_time_steps)
        self.path = None
        self.s_start_change = None
        self.i_start_change = None
        self.i_lane_change = None
        self.i_end_change = None

    def plane_acceleration_lane_path(self, street_info, ego_vp, s_start, dt, s_change_min_replan, ego_width, t_lane_change=4.):
        self.s_ref[0] = s_start
        self.d[0] = street_info.init_lane_path.get_d(s_start)
        self.path = MultiPath(street_info.init_lane_path)

        current_key = 0

        s_lc_min = max(street_info.s_lc_min, s_change_min_replan)
        s_start_change = s_lc_min + (street_info.s_lc_max - s_lc_min) * np.random.random_sample()

        self.s_start_change = s_start_change

        # Plan until s_start_change
        current_key = self.stay_until(s_start_change, ego_vp, street_info.init_lane_path, dt, current_key)
        self.i_start_change = current_key

        # Plan lane change
        if current_key < len(self.d):
            change_length = max(ego_vp.v[current_key] * t_lane_change, 5.)
            s_end_change = s_start_change + change_length
            current_key = self.change_lane(s_start_change, s_end_change, ego_vp, street_info, dt, current_key)

        # Plan after LaneChange
        self.i_end_change = current_key
        if current_key < len(self.d):
            self.stay(ego_vp, street_info.target_lane_path, dt, current_key)

        self.i_lane_change = self.determine_t_coop_lc(ego_width)

    def stay_until(self, s_max, ego_vp, path, dt, key):
        max_key = len(self.d)-1
        while self.s_ref[key] < s_max and key < max_key:
            s_next, d_next = path.calc_arc_coord_next(ego_vp.v[key], ego_vp.a[key], dt, self.s_ref[key])
            key += 1
            self.s_ref[key] = s_next
            self.d[key] = d_next

        return key

    def change_lane(self, s_start_change, s_end_change, ego_vp, street_info, dt, key):
        # complete multiple path
        cl_func = Polynomial_funktion.Poly5(s_start_change, street_info.init_lane_path.get_d(s_start_change),
                                            s_end_change, street_info.target_lane_path.get_d(s_end_change), 0., 0., 0.,
                                            0.).value

        lane_change_path = Path(cl_func, street_info)
        self.path.transition_path = lane_change_path
        self.path.target_path = street_info.target_lane_path
        self.path.s_init_to_trans = s_start_change
        self.path.s_trans_to_target = s_end_change

        # plan until change end
        max_key = len(self.d) -1
        while self.s_ref[key] < s_end_change and key < max_key:
            s_next, d_next = lane_change_path.calc_arc_coord_next(ego_vp.v[key], ego_vp.a[key], dt, self.s_ref[key])
            key += 1
            self.s_ref[key] = s_next
            self.d[key] = d_next

        return key

    def stay(self, ego_vp, path, dt, key):
        max_key = len(self.d)
        for i in range(key, max_key-1):
            s_next, d_next = path.calc_arc_coord_next(ego_vp.v[key], ego_vp.a[key], dt, self.s_ref[key])
            key += 1
            self.s_ref[key] = s_next
            self.d[key] = d_next

    def determine_t_coop_lc(self, ego_width):
        i_lane_change = 0
        for i in range(len(self.d)):
            i_lane_change = i
            if abs(self.d[i]) - ego_width/2. < 0:
                break

        return i_lane_change