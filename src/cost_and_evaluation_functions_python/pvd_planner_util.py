import numpy as np
import warnings
warnings.filterwarnings("error")


class VehicleState(object):
    def __init__(self, a, v, s_ref, d):
        self.a = a
        self.v = v
        self.s_ref = s_ref
        self.d = d


class TrajectorySet(object):
    def __init__(self, v, a, s, d, dt):
        self.v = v
        self.a = a
        self.s_ref = s
        self.d = d
        self.dt = dt

    def get_veh_trajectory(self, veh_key):
        return Trajectory(self.a[veh_key, :], self.v[veh_key, :], self.s_ref[veh_key, :], self.d[veh_key, :])

    def get_veh_state(self, veh_key, t_key):
        return VehicleState(self.a[veh_key, t_key], self.v[veh_key, t_key], self.s_ref[veh_key, t_key], self.d[veh_key, t_key])

    def change_trajectory_from_to(self, veh_key, c_from, c_to, trajectory):
        self.s_ref[veh_key, c_from:c_to] = trajectory.s_ref
        self.d[veh_key, c_from:c_to] = trajectory.d
        self.a[veh_key, c_from:c_to] = trajectory.a
        self.v[veh_key, c_from:c_to] = trajectory.v

    def get_trajectory_from_to(self, veh_key, get_from, to):
        a = self.a[veh_key, get_from:to]
        v = self.v[veh_key, get_from:to]
        s_ref = self.s_ref[veh_key, get_from:to]
        d = self.d[veh_key, get_from:to]

        return Trajectory(a, v, s_ref, d)


class SimulationParameter(object):
    def __init__(self, t_replan, sampling_range_t, planing_horizon, number_of_samples,
                 lc_duration, t_start_coop_before_lc):
        self.t_replan = t_replan
        self.planning_horizon = planing_horizon
        self.sampling_range_t = sampling_range_t
        self.num_of_samples = number_of_samples

        self.num_of_time_steps = int(round(planing_horizon/sampling_range_t))
        self.time_steps_in_planing_step = int(round(t_replan/sampling_range_t))

        self.lc_duration = lc_duration
        self.t_start_coop_before_ls = t_start_coop_before_lc


class TrajectoryXYYaw(object):
    def __init__(self):
        self.x_pos = None
        self.y_pos = None
        self.yaw = None

    def create_trajectory_from_arc_coord(self, s_ref, d_ref, path, street_info):
        self.x_pos = np.empty_like(s_ref)
        self.y_pos = np.empty_like(s_ref)
        self.yaw = np.empty_like(s_ref)

        for i in range(len(s_ref)):

            x, y = street_info.get_xy_from_sd(s_ref[i], d_ref[i])
            self.x_pos[i] = x
            self.y_pos[i] = y
            self.yaw[i] = path.get_yaw(s_ref[i])


class VelocityProfile(object):
    def __init__(self, a, v):
        self.a = a
        self.v = v


class Trajectory(object):
    def __init__(self, a, v, s_ref, d):
        self.a = a
        self.v = v
        self.s_ref = s_ref
        self.d = d


class TrajectorySamples(object):
    def __init__(self, a, v, s_ref, d_ref):
        self.a = a
        self.v = v
        self.s_ref = s_ref
        self.d = d_ref

    def get_trajectory(self, i):
        trajectory = Trajectory(self.a[i, :], self.v[i, :], self.s_ref[i, :], self.d[i, :])
        return trajectory


class CreateVelocityProfiles(object):
    def __init__(self, number_of_vps, sampling_horizon, sampling_range_t):
        self.number_of_vps = number_of_vps
        self.num_of_time_steps = int(round(sampling_horizon / sampling_range_t))

        self.a_matrix = np.empty([self.number_of_vps, self.num_of_time_steps], dtype=float)
        self.v_matrix = np.empty_like(self.a_matrix, dtype=float)

        self.dt = sampling_range_t

    def random_vp_by_jerk_given_start_state(self, v_start, a_start, v_max, v_min=0, a_min=-10, a_max=10, j_min_max=2):

        v_matrix, a_matrix = self.random_vp_by_jerk(self.num_of_time_steps, self.number_of_vps, v_start, a_start, v_max,
                                                    v_min, a_min, a_max, j_min_max)

        self.a_matrix = a_matrix
        self.v_matrix = v_matrix

    def random_vp_by_jerk_given_start_vp(self, start_vp, v_max, v_min=0, a_min=-10, a_max=10, j_min_max=2):
        self.a_matrix[:, 0:len(start_vp.a)] = start_vp.a
        self.v_matrix[:, 0:len(start_vp.v)] = start_vp.v

        a_start_remain = start_vp.a[-1]
        v_start_remain = start_vp.v[-1]
        num_time_steps_remain = self.num_of_time_steps - len(start_vp.a)
        v_remain, a_remain = self.random_vp_by_jerk(num_time_steps_remain, self.number_of_vps, v_start_remain,
                                                    a_start_remain, v_max, v_min, a_min, a_max, j_min_max)

        self.a_matrix[:, len(start_vp.a):] = a_remain
        self.v_matrix[:, len(start_vp.a):] = v_remain

    def random_vp_by_jerk(self, num_of_time_steps, num_of_vps, v_start, a_start, v_max, v_min, a_min, a_max, j_min_max):
        a_matrix_t = np.empty([num_of_time_steps, num_of_vps], dtype=float)
        v_matrix_t = np.empty_like(a_matrix_t, dtype=float)

        a_matrix_t[0, :] = a_start
        v_matrix_t[0, :] = v_start

        j_matrix = (np.random.rand(num_of_time_steps, num_of_vps) - 0.5) * j_min_max * 2

        for j in range(num_of_time_steps - 1):
            a_row = a_matrix_t[j + 1, :]
            a_last_row = a_matrix_t[j, :]

            # Calculate acceleration for this time stamp with random jerks. Limit acceleration to a_max/a_min if to high/low
            a_row[:] = a_last_row + j_matrix[j, :] * self.dt
            a_row[a_row > a_max] = a_max
            a_row[a_row < a_min] = a_min

            # Calculate velocity for this time stamp with acceleration. Limit velocity to v_max/v_min if to high/low
            # Recalculate acceleration for last timestamps where velocity has been restricted
            v_row = v_matrix_t[j + 1, :]
            v_last_row = v_matrix_t[j, :]
            v_row[:] = v_last_row + a_last_row * self.dt
            mask_max = v_row > v_max
            mask_min = v_row < v_min
            v_row[mask_max] = v_max
            v_row[mask_min] = v_min
            mask = np.logical_or(mask_min, mask_max)
            a_last_row[mask] = (v_row[mask] - v_last_row[mask]) / self.dt

        a_matrix_t[num_of_time_steps - 1, :] = 0.

        return np.transpose(v_matrix_t), np.transpose(a_matrix_t)

    def get_velocity_profile(self, i):
        vp = VelocityProfile(self.a_matrix[i, :], self.v_matrix[i, :])
        return vp


def random_trajectories(number_of_samples, planing_horizon, dt, v_start, a_start, s_start, v_max, path, v_min=0,
                        a_min=-10, a_max=10):
    at_matrix, v_matrix = random_vp(number_of_samples, planing_horizon, dt, v_start, a_start, v_max, v_min, a_min, a_max)
    s_ref_matrix, d_matrix = calc_arc_coor_matrix(at_matrix, v_matrix, s_start, planing_horizon, dt, path)
    trajectory_samples = TrajectorySamples(at_matrix, v_matrix, s_ref_matrix, d_matrix)
    return trajectory_samples


def random_vp(number_of_samples, sampling_horizon, sampling_range_t, v_start, a_start, v_max, v_min=0, a_min=-10,
              a_max=10):
    """

    :param number_of_samples: number of velocity profile samples
    :param sampling_horizon: planning horizon in seconds
    :param sampling_range_t: time gap between two time stamps
    :param v_start: velocity at start of planning
    :param a_start: acceleration at start of planning
    :param v_max: max velocity
    :param v_min: min velocity
    :param a_min: min acceleration
    :param a_max: max acceleration
    :return:
    """
    row_size_r = number_of_samples
    column_size_r = int(round(sampling_horizon / sampling_range_t))

    # Create matrix with random jerks between -2 and 2
    j_matrix = (np.random.rand(column_size_r, row_size_r) - 0.5) * 4

    a_matrix = np.empty_like(j_matrix)
    v_matrix = np.empty_like(j_matrix)

    a_matrix[0, :] = a_start
    v_matrix[0, :] = v_start

    for j in range(column_size_r - 1):
        a_row = a_matrix[j + 1, :]
        a_last_row = a_matrix[j, :]

        # Calculate acceleration for this time stamp with random jerks. Limit acceleration to a_max/a_min if to high/low
        a_row[:] = a_last_row + j_matrix[j, :] * sampling_range_t
        a_row[a_row > a_max] = a_max
        a_row[a_row < a_min] = a_min

        # Calculate velocity for this time stamp with acceleration. Limit velocity to v_max/v_min if to high/low
        # Recalculate acceleration for last timestamps where velocity has been restricted
        v_row = v_matrix[j + 1, :]
        v_last_row = v_matrix[j, :]
        v_row[:] = v_last_row + a_last_row * sampling_range_t
        mask_max = v_row > v_max
        mask_min = v_row < v_min
        v_row[mask_max] = v_max
        v_row[mask_min] = v_min
        mask = np.logical_or(mask_min, mask_max)
        a_last_row[mask] = (v_row[mask] - v_last_row[mask]) / sampling_range_t

    a_matrix[column_size_r-1, :] = 0.
    a_matrix = np.transpose(a_matrix)
    v_matrix = np.transpose(v_matrix)

    return a_matrix, v_matrix


def calc_arc_coor_matrix(at_matrix, v_matrix, s_start, planing_horizon, dt, path):
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