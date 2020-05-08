import scenario_acceleration_lane_street
import Vehicle
import pvd_planner_util
import Street
import prediction
import find_cooperative_trajectories
from operator import attrgetter
from scipy import interpolate
#import lanelet2
import numpy as np
import time
import replaning
import matplotlib.pyplot as plt


def cooperative_lane_change(vehicle_list, start_vp, street_info, sim_para):
    # Generate random velocity profiles for ego
    ego_vps = pvd_planner_util.CreateVelocityProfiles(sim_para.num_of_samples, sim_para.planning_horizon, sim_para.sampling_range_t)
    ego_vps.random_vp_by_jerk_given_start_vp(start_vp, vehicle_list.veh_list[0].v_max)

    # Generate ego independent prediction for all relevant vehicle
    veh_prediction = prediction.LaneBasedPrediction(vehicle_list.num_of_veh_tl+vehicle_list.num_of_veh_il,
                                                    sim_para)
    veh_prediction.idm_prediction(vehicle_list)

    # For every single vp find best cooperation
    solutions = []
    for i in range(sim_para.num_of_samples):
        if i%100 == 0:
            print "Step", i
        solution = find_cooperative_trajectories.CooperativeSolution(veh_prediction, ego_vps.get_velocity_profile(i),
                                                                     len(vehicle_list.veh_list), sim_para)
        solution.acceleration_lane_planer(street_info, vehicle_list, ego_vps.get_velocity_profile(i))
        solutions.append(solution)

    # Select best solution
    best_solution = min(solutions, key=attrgetter('cost'))

    print "cost best solution", best_solution.cost

    return best_solution


def single_shoot(vehicle_list, ego_a_start_list, ego_v_start_list, street_info, t_replanning, sampling_range_t=0.5,
                 planning_horizon=15., num_of_samples=1000):
    # generate trajectories with cooperative planner
    solution = cooperative_lane_change(vehicle_list, ego_a_start_list, ego_v_start_list, street_info, t_replanning,
                                       sampling_range_t, planning_horizon, num_of_samples)
    vehicle_list.veh_list[0].path = solution.ego_path

    trajectories_for_ros = []
    for i in range(len(vehicle_list.veh_list)):
        trajectory_ros = pvd_planner_util.TrajectoryXYYaw()
        trajectory_ros.create_trajectory_from_arc_coord(solution.s_ref[i], solution.d[i], vehicle_list.veh_list[i].path, street_info)
        trajectories_for_ros.append(trajectory_ros)

    return trajectories_for_ros


def ros_cooperative_lane_change(center_line, init_lane, target_lane, veh_s_start_list, veh_id_list,
                                sampling_range_t=0.5, planning_horizon=15, num_of_samples=1000, t_replan = 1):
    print "Starting Cooperative Planner"
    # Init street_info
    v_max = 50/3.6
    s_lc_min = 100.
    s_lc_max = 220.

    s_init = np.empty(len(init_lane))
    d_init = np.empty(len(init_lane))
    for i in range(len(init_lane)):
        xy = lanelet2.core.BasicPoint2d(init_lane[i].x, init_lane[i].y)
        arc = lanelet2.geometry.toArcCoordinates(center_line, xy)
        s_init[i] = arc.length
        d_init[i] = arc.distance

    init_lane_func = interpolate.interp1d(s_init, d_init)

    s_target = np.empty(len(target_lane))
    d_target = np.empty(len(target_lane))
    for i in range(len(target_lane)):
        xy = lanelet2.core.BasicPoint2d(target_lane[i].x, target_lane[i].y)
        arc = lanelet2.geometry.toArcCoordinates(center_line, xy)
        s_target[i] = arc.length
        d_target[i] = arc.distance

    target_lane_func = interpolate.interp1d(s_target, d_target)
    street_info = Street.StreetInfoROS(center_line, init_lane_func, target_lane_func, v_max, s_lc_min, s_lc_max)

    # init veh
    # init ego veh
    ego_v_start = 48./3.6
    ego_v_opt = 48./3.6
    ego_v_max = 50./3.6
    ego_a_start = 0.
    ego_xy = lanelet2.geometry.interpolatedPointAtDistance(init_lane, veh_s_start_list[0])
    ego_arc_coord = lanelet2.geometry.toArcCoordinates(center_line, ego_xy)
    ego_s_ref_start = ego_arc_coord.length
    ego_d_start = ego_arc_coord.distance
    vehicle_list = Vehicle.VehicleList
    ego_veh = Vehicle.Vehicle(veh_id_list[0], ego_s_ref_start, ego_d_start, ego_v_start, ego_v_opt, ego_v_max,
                              sampling_range_t, ego_a_start)
    ego_veh.path = street_info.init_lane_path
    vehicle_list.veh_list.append(ego_veh)

    ego_a_start_list = [ego_a_start]
    ego_v_start_list = [ego_v_start]
    for i in range(int(round(t_replan/sampling_range_t))-1):
        ego_a_start_list.append(0.)
        ego_v_start_list.append(ego_v_start_list[0])

    # init other vehicle
    v_start = 48. / 3.6
    v_opt = 48. / 3.6
    v_max = 50. / 3.6
    a_start = 0.
    for i in range(1, len(veh_id_list)):
        veh_xy = lanelet2.geometry.interpolatedPointAtDistance(target_lane, veh_s_start_list[i])
        veh_arc_coord = lanelet2.geometry.toArcCoordinates(center_line, veh_xy)
        s_ref_start = veh_arc_coord.length
        d_start = veh_arc_coord.distance
        add_veh = Vehicle.Vehicle(veh_id_list[i], s_ref_start, d_start, v_start, v_opt, v_max, sampling_range_t, a_start)
        add_veh.path = street_info.target_lane_path
        vehicle_list.num_of_veh_tl += 1
        vehicle_list.veh_list.append(add_veh)

    trajectories_for_ros = replaning.replanning_multi_agent_optimum(vehicle_list, ego_a_start_list, ego_v_start_list, street_info,
                                                          t_replan, sampling_range_t, planning_horizon, num_of_samples)

    return trajectories_for_ros


if __name__ == "__main__":
    # Create a sample street for the lane change scenario
    # In this case one acceleration lane and one main lane
    street = scenario_acceleration_lane_street.create_road_with_acceleration_lane(s_length=300., number_of_lanes=1,
                                                                                  lane_width=3.5, s_change_start=120.,
                                                                                  s_change_end=170., v_max=50./3.6)

    # Creating fake init lane
    s_lane = np.empty(len(range(0, 500)))
    d_init_lane = np.empty_like(s_lane)
    for i in range(0, 500):
        s_lane[i] = i
        d_init_lane[i] = -1.75

    d_target_lane = np.empty_like(s_lane)
    for i in range(0, 500):
        d_target_lane[i] = 1.75

    init_lane_func = interpolate.interp1d(s_lane, d_init_lane)
    target_lane_func = interpolate.interp1d(s_lane, d_target_lane)

    street_info = Street.LaneChangeStreetInfo_fake(street.lane_markers[1], init_lane_func, target_lane_func, 50./3.6,
                                                   120., 170.)

    # simulation parameter
    sim_para = pvd_planner_util.SimulationParameter(t_replan=1., sampling_range_t=0.2, planing_horizon=15,
                                                    number_of_samples=1000, lc_duration=4., t_start_coop_before_lc=6.5)

    # Init vehicle list
    vehicle_list = Vehicle.VehicleList

    # Init ego vehicle
    ego_s_start = 90.
    ego_d_start = -1.75
    ego_v_start = 45./3.6
    ego_a_start = 0
    ego_v_opt = 50./3.6
    ego_v_max = 50./3.6
    ego_veh_index = 0

    t_replaning = 1.
    ego_a_start_list = [0., 0., 0., 0., 0.]
    ego_v_start_list = [45./3.6, 45./3.6, 45./3.6, 45./3.6, 45./3.6]

    ego_start_vp = pvd_planner_util.VelocityProfile(ego_a_start_list, ego_v_start_list)

    ego_veh = Vehicle.Vehicle(ego_veh_index, ego_s_start, ego_d_start, ego_v_start, ego_v_opt, ego_v_max,
                              sim_para.sampling_range_t, ego_a_start)
    ego_veh.path = street_info.init_lane_path
    vehicle_list.veh_list.append(ego_veh)

    # Init target lane vehicle
    num_of_tl_veh = 4
    tl_s_start = [140., 110., 80., 50.]
    tl_d_start = [1.75, 1.75, 1.75, 1.75]
    tl_v_start = [(48./3.6), (48./3.6), (48./3.6), (48./3.6)]
    tl_a_start = [0.1, -0.1, 0.1, -0.1]
    tl_v_opt = [(48./3.6), (48./3.6), (48./3.6), (48./3.6)]
    tl_v_max = [50./3.6, 50./3.6, 50./3.6, 50./3.6]
    tl_veh_index = [1, 2, 3, 4]

    for i in range(num_of_tl_veh):
        add_veh = Vehicle.Vehicle(tl_veh_index[i], tl_s_start[i], tl_d_start[i], tl_v_start[i], tl_v_opt[i], tl_v_max[i],
                                  sim_para.sampling_range_t, tl_a_start[i])
        add_veh.path = street_info.target_lane_path
        vehicle_list.num_of_veh_tl += 1
        vehicle_list.veh_list.append(add_veh)

    replan_solution = replaning.Replanning(sim_para, vehicle_list, ego_start_vp, street_info)
    replan_solution.multi_agent_optimum_replan()
    print 'wait'

    plt.figure()

    ax_s = plt.subplot2grid((13, 1), (0, 0), rowspan=4)
    ax_d = plt.subplot2grid((13, 1), (5, 0), rowspan=2, sharex=ax_s)
    ax_v = plt.subplot2grid((13, 1), (8, 0), rowspan=2, sharex=ax_s)
    ax_a = plt.subplot2grid((13, 1), (11, 0), rowspan=2, sharex=ax_s)

    t_list = np.empty(len(replan_solution.trajectory_set.v[0]))
    for i in range(len(t_list)):
        t_list[i] = i*sim_para.sampling_range_t
    for i in range(len(replan_solution.trajectory_set.v)):
        ax_s.plot(t_list, replan_solution.trajectory_set.s_ref[i])
        ax_d.plot(t_list, replan_solution.trajectory_set.d[i])
        ax_v.plot(t_list, replan_solution.trajectory_set.v[i])
        ax_a.plot(t_list, replan_solution.trajectory_set.a[i])

    plt.xlabel('t [s]')
    ax_s.set_ylabel('s [m]')
    ax_d.set_ylabel('d [m]')
    ax_v.set_ylabel('v [m/s]')
    ax_a.set_ylabel('a [m/s^2]')

    plt.show()




    # t_start = time.time()
    # solution = cooperative_lane_change(vehicle_list, ego_a_start_list, ego_v_start_list, street_info, t_replaning,
    #                                    sampling_range_t, planning_horizon, number_of_samples)
    #
    # plt.figure()
    #
    # ax_s = plt.subplot2grid((13, 1), (0, 0), rowspan=4)
    # ax_d = plt.subplot2grid((13, 1), (5, 0), rowspan=2, sharex=ax_s)
    # ax_v = plt.subplot2grid((13, 1), (8, 0), rowspan=2, sharex=ax_s)
    # ax_a = plt.subplot2grid((13, 1), (11, 0), rowspan=2, sharex=ax_s)
    #
    #
    # t_list = np.empty(np.size(solution.v, axis=1))
    # for i in range(len(t_list)):
    #     t_list[i] = i*sampling_range_t
    # for i in range(np.size(solution.v, axis=0)):
    #     ax_s.plot(t_list, solution.s_ref[i, :])
    #     ax_d.plot(t_list, solution.d[i, :])
    #     ax_v.plot(t_list, solution.v[i, :])
    #     ax_a.plot(t_list, solution.a[i, :])
    #
    # plt.xlabel('t [s]')
    # ax_s.set_ylabel('s [m]')
    # ax_d.set_ylabel('d [m]')
    # ax_v.set_ylabel('v [m/s]')
    # ax_a.set_ylabel('a [m/s^2]')
    #
    # plt.show()

    #t_end = time.time()
    #print "clac time: ", t_end - t_start
