import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Vehicle
import math
import general_functions as g_func
import scenario_acceleration_lane_street
import safety_distance as sd
import time
# import cv2

ego_i = 0

class VehicleOutlines(object):
    """
    This class contains the four points that describe the box of the vehicle outlines
    """
    def __init__(self, s, x_reference, y_reference, path, veh_properties):
        """
        :type s: float
        :param s: current position of the veh in frenet coordinates
        :type x_reference: function
        :param x_reference: x(s) of the reference curve
        :type y_reference: function
        :param y_reference: y(s) of the reference curve
        :type path: function
        :param path: vehicle path in in frenet coordinates d(s)
        :type veh_properties: Vehicle.VehicleProperties
        :param veh_properties: properties of the vehicle
        """
        x, y = g_func.frenet_to_world_coordinates(x_reference, y_reference, s, path(s))
        heading_angle_street = g_func.calc_heading_angle(x_reference, y_reference, s)
        path_d = (path(s + 0.1) - path(s - 0.1)) / (2 * 0.1)

        heading_angle_path = np.arctan(path_d)
        heading_angle = heading_angle_path + heading_angle_street

        r1 = ((veh_properties.width/2)**2 + veh_properties.length_to_front_bumper**2)**0.5
        phi1 = np.arctan((veh_properties.width/2)/veh_properties.length_to_front_bumper)
        self.y1 = y + np.sin(heading_angle+phi1) * r1
        self.x1 = x + np.cos(heading_angle+phi1) * r1
        self.y2 = y + np.sin(heading_angle-phi1) * r1
        self.x2 = x + np.cos(heading_angle-phi1) * r1

        r2 = ((veh_properties.width/2)**2 + veh_properties.length_to_rear_bumper**2)**0.5
        phi2 = np.arctan((veh_properties.width/2)/veh_properties.length_to_rear_bumper)
        self.y4 = y - np.sin(heading_angle-phi2) * r2
        self.x4 = x - np.cos(heading_angle-phi2) * r2
        self.y3 = y - np.sin(heading_angle+phi2) * r2
        self.x3 = x - np.cos(heading_angle+phi2) * r2


def counter():
    if 'cnt' not in counter.__dict__:
        counter.cnt = 0
    counter.cnt += 1
    return counter.cnt


def create_data_files_for_street_visualisation(street, output_directory=''):
    """
    :type street: Street.Street
    :param street: road that is going to be visualized
    :type output_directory: string
    :param output_directory: directory where the data files are going to be saved
    :return: ---
    """
    # Create one data file for every lane_marker
    for i in range(len(street.lane_markers)):
        # Create string for output directory
        file_name = output_directory + '/l' + str(i) + '.dat'
        f = open(file_name, "w")

        # Write first lane of the file
        f.write('#X   Y \n')
        s_list = np.arange(street.lane_markers[i].s_start, street.lane_markers[i].s_end, 0.1)

        # Write data in to file
        for s in s_list:
            # Convert position in frenet coordinates in to position in global coordinates
            x, y = g_func.frenet_to_world_coordinates(street.reference_curve[0], street.reference_curve[1], s,
                                               street.lane_markers[i].value(s))
            f.write(str(x) + '   ' + str(y) + '\n')
        f.close()


def plot_scenario_vehicle_list(vehicle_list, t_start, t_end, sampling_range_t, t_list, street, delta_t=0.01, output_directory=''):
    """
    This function creates datafiles to plot a timed map
    :return: None
    """

    # Create data files to visualise the street
    create_data_files_for_street_visualisation(street, output_directory)

    # Create lists to plot vehicles for visualisation
    ego_plot_list = []
    cooperative_plot_list = []
    other_plot_list = []

    for vehicle in vehicle_list:
        vehicle_plot_list = create_plot_list(vehicle, t_start, t_end, sampling_range_t, t_list, delta_t)
        if vehicle.typ == 0:
            create_file_to_plot_path(vehicle, output_directory)
            ego_plot_list.append(vehicle_plot_list)
        elif vehicle.typ == 1:
            cooperative_plot_list.append(vehicle_plot_list)
        else:
            other_plot_list.append(vehicle_plot_list)

    # Create data files to visualise vehicles
    if ego_plot_list:
        create_data_files_for_plot_lists(ego_plot_list, output_directory, 'ego')
    if cooperative_plot_list:
        create_data_files_for_plot_lists(cooperative_plot_list, output_directory, 'cop')
    if other_plot_list:
        create_data_files_for_plot_lists(other_plot_list, output_directory, 'others')


def create_file_to_plot_path(vehicle, output_directory):
    """
    :type vehicle: Vehicle.Vehicle
    :param vehicle:
    :param output_directory:
    :return:
    """
    s_start = vehicle.s_start
    s_end = vehicle.trajectory.s_reference_list[-1]
    file_name = output_directory + '/ego_path.dat'
    f = open(file_name, "w")

    # Write first lane of the file
    f.write('#X   Y \n')
    s_list = np.arange(s_start, s_end, 0.1)

    # Write data in to file
    for s in s_list:
        # Convert position in frenet coordinates in to position in global coordinates
        x, y = g_func.frenet_to_world_coordinates(vehicle.trajectory.street.reference_curve[0],
                                                  vehicle.trajectory.street.reference_curve[1], s,
                                                  vehicle.trajectory.path(s))
        f.write(str(x) + '   ' + str(y) + '\n')

    f.close()


def create_plot_list(vehicle, t_start, t_end, sampling_range_t, t_list, delta_t):
    """
    :type vehicle: Vehicle.Vehicle
    :param vehicle: vehicle to plot
    :param t_start: start_time of visualisation
    :param t_end: end_time of visualisation
    :param delta_t: time between two frames
    :return:
    """
    street = vehicle.trajectory.street

    samples_until_start = int((max(t_list[0]-t_start, 0.))/delta_t)
    samples_after_end = int((max(t_end-t_list[-1], 0.))/delta_t)
    start_key = int((max(t_start-t_list[0], 0.))/sampling_range_t)
    end_key = min(int((t_end - t_list[0]) / sampling_range_t),
                  int((t_list[-1] - t_list[0]) / sampling_range_t))
    plot_list = [None] * samples_until_start

    for i in range(start_key, end_key):
        delta_s_reference = vehicle.trajectory.s_reference_list[i+1] - vehicle.trajectory.s_reference_list[i]
        delta_s_track = vehicle.trajectory.v_list[i] * sampling_range_t + 0.5 * vehicle.trajectory.at_list[
            i] * sampling_range_t * sampling_range_t

        s_r = vehicle.trajectory.s_reference_list[i]
        v = vehicle.trajectory.v_list[i]
        a = vehicle.trajectory.at_list[i]

        frames = int(math.ceil(sampling_range_t / delta_t))
        for j in range(frames):
            s_r += g_func.calc_ds(delta_t, v, a) * (delta_s_reference/delta_s_track)
            v += a*delta_t
            plot_list.append(VehicleOutlines(s_r, street.reference_curve[0], street.reference_curve[1], vehicle.trajectory.path, vehicle.properties))

    plot_list += [None]*samples_after_end

    return plot_list


def create_data_files_for_plot_lists(plot_list, output_directory, filename):
    """
    :type plot_list: list
    :param plot_list:
    :type output_directory: str
    :param output_directory: output directory
    :type filename: str
    :param filename: string that is in front of the number of the filename
    :return:
    """
    for i in range(len(plot_list[0])):
        file_name = output_directory + '/' + filename + str(i) + '.dat'
        f = open(file_name, "w")
        line_string = '#'
        for j in range(len(plot_list)):
            line_string += 'X   Y  '
        line_string += '\n'
        f.write(line_string)

        line_string = ''
        for j in range(len(plot_list)):
            try:
                line_string += str(plot_list[j][i].x1) + '   ' + str(plot_list[j][i].y1) + '   '
            except AttributeError:
                line_string += 'NaN' + '   ' + 'NaN' + '   '
        line_string += '\n'
        f.write(line_string)

        line_string = ''
        for j in range(len(plot_list)):
            try:
                line_string += str(plot_list[j][i].x2) + '   ' + str(plot_list[j][i].y2) + '   '
            except AttributeError:
                line_string += 'NaN' + '   ' + 'NaN' + '   '
        line_string += '\n'
        f.write(line_string)

        line_string = ''
        for j in range(len(plot_list)):
            try:
                line_string += str(plot_list[j][i].x3) + '   ' + str(plot_list[j][i].y3) + '   '
            except AttributeError:
                line_string += 'NaN' + '   ' + 'NaN' + '   '
        line_string += '\n'
        f.write(line_string)

        line_string = ''
        for j in range(len(plot_list)):
            try:
                line_string += str(plot_list[j][i].x4) + '   ' + str(plot_list[j][i].y4) + '   '
            except AttributeError:
                line_string += 'NaN' + '   ' + 'NaN' + '   '
        line_string += '\n'
        f.write(line_string)

        line_string = ''
        for j in range(len(plot_list)):
            try:
                line_string += str(plot_list[j][i].x1) + '   ' + str(plot_list[j][i].y1) + '   '
            except AttributeError:
                line_string += 'NaN' + '   ' + 'NaN' + '   '
        line_string += '\n'
        f.write(line_string)

        f.close()


def sampling_visualistion(v_matrix, at_matrix, t_list, dt):

    s_matrix_ds = g_func.calc_ds(dt, v_matrix, at_matrix)
    s_matrix = np.empty_like(s_matrix_ds)
    s_matrix[:, 0] = 0
    for i in range(np.size(s_matrix_ds, axis=1)-1):
        s_matrix[:, i+1] = s_matrix[:, i] + s_matrix_ds[:, i]
    plt.figure()

    ax_s = plt.subplot2grid((10, 1), (0, 0), rowspan=4)
    ax_v = plt.subplot2grid((10, 1), (5, 0), rowspan=2, sharex=ax_s)
    ax_a = plt.subplot2grid((10, 1), (8, 0), rowspan=2, sharex=ax_s)

    for i in range(np.size(v_matrix, axis=0)):
        ax_s.plot(t_list, s_matrix[i, :])
        ax_v.plot(t_list, v_matrix[i, :])
        ax_a.plot(t_list, at_matrix[i, :])

    plt.xlabel('t [s]')
    ax_s.set_ylabel('s [m]')
    ax_v.set_ylabel('v [m/s]')
    ax_a.set_ylabel('a [m/s^2]')

    plt.show()


def visualisation_trajectories(sample_list, ego_veh, time_of_cooperation, title):

    s_min_list = sample_list[0].trajectory.s_reference_list
    s_max_list = sample_list[0].trajectory.s_reference_list
    for sample in sample_list:
        s_min_list = np.minimum(s_min_list, sample.trajectory.s_reference_list)
        s_max_list = np.maximum(s_max_list, sample.trajectory.s_reference_list)

    v_min_list = sample_list[0].trajectory.v_list
    v_max_list = sample_list[0].trajectory.v_list
    for sample in sample_list:
        v_min_list = np.minimum(v_min_list, sample.trajectory.v_list)
        v_max_list = np.maximum(v_max_list, sample.trajectory.v_list)

    s_max = max(s_max_list)
    s_min = min(s_min_list)

    sd_max = s_max_list + v_max_list*1.8
    sd_min = s_min_list + v_min_list*1.8
    sd_ego = ego_veh.trajectory.s_reference_list + np.asarray(ego_veh.trajectory.v_list)*1.8

    plt.figure()
    plt.title(title)
    plt.fill_between(sample_list[0].trajectory.t_list, sd_min, s_min_list, facecolor='blue', alpha=0.2)
    plt.fill_between(sample_list[0].trajectory.t_list, sd_max, s_max_list, facecolor='blue', alpha=0.4)
    plt.fill_between(sample_list[0].trajectory.t_list, ego_veh.trajectory.s_reference_list, sd_ego, facecolor='red', alpha=0.3)
    plt.plot(sample_list[0].trajectory.t_list, s_min_list, 'b')
    plt.plot(sample_list[0].trajectory.t_list, s_max_list, 'b')
    plt.plot(sample_list[0].trajectory.t_list, ego_veh.trajectory.s_reference_list, 'r')
    plt.plot([time_of_cooperation, time_of_cooperation], [s_min, s_max], 'r')
    plt.show()


    for t in ego_veh.trajectory.cooperative_actions:
        print 'time of cooperation', t


def plot_best_cooperation(ego, partner, time_of_cooperation, title):

    s_max = max(ego.trajectory.s_reference_list)
    s_min = min(ego.trajectory.s_reference_list)

    t_list = ego.trajectory.t_list

    s_ego = ego.trajectory.s_reference_list
    sd_ego = s_ego + np.asarray(ego.trajectory.v_list)*1.8

    s_partner = partner.trajectory.s_reference_list
    sd_partner = s_partner + np.asarray(partner.trajectory.v_list) * 1.8

    plt.figure()
    plt.title(title)
    plt.fill_between(t_list, s_ego, sd_ego, facecolor='red', alpha=0.3)
    plt.fill_between(t_list, s_partner, sd_partner, facecolor='blue', alpha=0.3)
    plt.plot(t_list, s_ego, 'r')
    plt.plot(t_list, s_partner, 'b')
    plt.plot([time_of_cooperation, time_of_cooperation], [s_max, s_min], 'r')


def add_street_to_plot(ax, street, x_min, x_max):


    ax.set_aspect(1)
    ax.axis([x_min, x_max, -5, 5])

    for i in range(len(street.lane_markers)):
        s_list = np.arange(max(street.lane_markers[i].s_start, x_min), min(street.lane_markers[i].s_end, x_max), 0.1)

        x_list = np.empty_like(s_list)
        y_list = np.empty_like(s_list)
        # Write data in to file
        for j in range(len(s_list)):
            # Convert position in frenet coordinates in to position in global coordinates
            x, y = g_func.frenet_to_world_coordinates(street.reference_curve[0], street.reference_curve[1], s_list[j],
                                                      street.lane_markers[i].value(s_list[j]))
            x_list[j] = x
            y_list[j] = y

        if street.lane_markers[i].typ == 0:
            ax.plot(x_list, y_list, color='black', linewidth=2)
        else:
            ax.plot(x_list, y_list, color='black', linestyle='--', linewidth=2)


def add_vehicle_to_plot(ax, length, width, length_to_front_bumper, x_pos, y_pos, heading_angle, color):
    x_list = np.empty(5, dtype=float)
    y_list = np.empty(5, dtype=float)

    r1 = ((width / 2) ** 2 + length_to_front_bumper ** 2) ** 0.5
    phi1 = np.arctan((width / 2) / length_to_front_bumper)
    y_list[0] = y_pos + np.sin(heading_angle + phi1) * r1
    y_list[4] = y_pos + np.sin(heading_angle + phi1) * r1
    x_list[0] = x_pos + np.cos(heading_angle + phi1) * r1
    x_list[4] = x_pos + np.cos(heading_angle + phi1) * r1
    y_list[1] = y_pos + np.sin(heading_angle - phi1) * r1
    x_list[1] = x_pos + np.cos(heading_angle - phi1) * r1

    r2 = ((width / 2) ** 2 + (length-length_to_front_bumper) ** 2) ** 0.5
    phi2 = np.arctan((width / 2) / (length - length_to_front_bumper))
    y_list[3] = y_pos - np.sin(heading_angle - phi2) * r2
    x_list[3] = x_pos - np.cos(heading_angle - phi2) * r2
    y_list[2] = y_pos - np.sin(heading_angle + phi2) * r2
    x_list[2] = x_pos - np.cos(heading_angle + phi2) * r2

    ax.plot(x_list, y_list, color=color)
    ax.fill(x_list, y_list, color=color, alpha=0.5)


def unit_test_find_best_cooperation(samples_s, samples_v, ego_s, start_i, end_i, best_sample, i_coop, veh_i):
    fig, ax = plt.subplots()

    x_axes = np.zeros_like(ego_s[start_i: end_i])
    for i in range(len(x_axes)): x_axes[i] = i + start_i

    for j in range(np.size(samples_v,0)):
        ax.plot(x_axes, samples_s[j, :], color='yellow')
    ax.plot(x_axes, samples_s[best_sample, :], color='blue', linewidth = 2)


    ax.plot(x_axes, ego_s[start_i: end_i], color='red')
    ax.axvline(i_coop, color='green', linewidth= 2)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('s [m]')
    red_patch = mpatches.Patch(color='red', label='Ego Vehicle')
    orange_patch = mpatches.Patch(color='yellow', label='Vehicle Samples')
    blue_patch = mpatches.Patch(color='blue', label='Best Sample')
    green_patch = mpatches.Patch(color='green', label='Time of cooperation')
    #ax.legend(handles=[red_patch, orange_patch, green_patch, blue_patch], loc=2, prop={'size': 10})
    plt.title('Following Vehicle')
    plt.grid(True)
    #plt.show()
    figname = '/home/diestmann/Documents/post_pro/Unit_tests/Find_best_cooperation/' + str(int(time.time())) + '_' + str(veh_i)
    fig.savefig(figname)
    plt.close()


def unit_test_classification(ego_samples, i_coop, ego_veh, ego_i, vehicle_list, involved_list, affected_list, unaffected_list):
    street = scenario_acceleration_lane_street.create_road_with_acceleration_lane(s_length=200., number_of_lanes=1,
                                                                                  lane_width=3.5, s_change_start=40.,
                                                                                  s_change_end=80., v_max=50. / 3.6)
    fig, ax = plt.subplots(figsize=(20, 10))
    add_street_to_plot(ax, street, 30., 115.)

    #ego vehicle
    ego_x, ego_y = g_func.frenet_to_world_coordinates(street.reference_curve[0], street.reference_curve[1],
                                                      ego_samples.s_ref_matrix[ego_i, i_coop],
                                                      ego_samples.d_matrix[ego_i, i_coop])
    heading_angle = 0
    add_vehicle_to_plot(ax, ego_veh.properties.length, ego_veh.properties.width, ego_veh.properties.length_to_front_bumper, ego_x, ego_y, heading_angle, 'red')

    for j in range(len(vehicle_list)):
        veh_ind = ego_samples.get_index(ego_i, j)
        veh_x, veh_y = g_func.frenet_to_world_coordinates(street.reference_curve[0], street.reference_curve[1],
                                                          ego_samples.veh_sr_matrix[veh_ind, i_coop],
                                                          ego_samples.veh_d_matrix[veh_ind, i_coop])
        heading_angle = 0
        if j in involved_list:
            color = 'orange'
        elif j in affected_list:
            color = 'blue'
        else:
            color = 'green'

        add_vehicle_to_plot(ax, vehicle_list[j].properties.length, vehicle_list[j].properties.width, vehicle_list[j].properties.length_to_front_bumper, veh_x, veh_y, heading_angle, color)

    figname = 'Unit_tests/Classification/' + str(ego_i)
    ax.set_xlabel('d [m]')
    ax.set_ylabel('s [m]')
    red_patch = mpatches.Patch(color='red', label='Ego Vehicle')
    orange_patch = mpatches.Patch(color='orange', label='Involved')
    green_patch = mpatches.Patch(color='green', label='Unaffected')
    blue_patch = mpatches.Patch(color='blue', label='Affected')
    ax.legend(handles=[red_patch, orange_patch, blue_patch, green_patch])
    fig.savefig(figname)
    plt.close()


def unit_test_path_planing(ego_vehicle, path_list, until_list, street, ego_i):
    fig, ax = plt.subplots(figsize=(20, 10))
    add_street_to_plot(ax, street, 30., 115.)
    x, y = g_func.frenet_to_world_coordinates(street.reference_curve[0], street.reference_curve[1], ego_vehicle.s_start, ego_vehicle.d_start)

    heading_angle = 0

    add_vehicle_to_plot(ax, ego_vehicle.properties.length, ego_vehicle.properties.width, ego_vehicle.properties.length_to_front_bumper, x, y, heading_angle, 'red')

    s_start = until_list[0]
    s_end = until_list[0 + 1]
    s_list_i = np.arange(s_start, s_end, 0.1)
    d_list_i = path_list[0](s_list_i)
    ax.plot(s_list_i, d_list_i, color='red', linewidth=1.5, label='inital lane')

    s_start = until_list[1]
    s_end = until_list[1 + 1]
    s_list_c = np.arange(s_start, s_end, 0.1)
    d_list_c = path_list[1](s_list_c)
    ax.plot(s_list_c, d_list_c, color='green', linestyle='--', label='lane change', linewidth=1.5)


    s_start = until_list[2]
    s_end = until_list[2 + 1]
    s_list_t = np.arange(s_start, s_end, 0.1)
    d_list_t = path_list[2](s_list_t)
    ax.plot(s_list_t, d_list_t, color='blue', label='target lane', linewidth=1.5)

    ax.plot(s_list_c[0], d_list_c[0], 'rs', label='start_change')
    ax.plot(s_list_c[-1], d_list_c[-1], 'gs', label='start_end')

    ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    figname = 'Unit_tests/PathPlanning/' + str(ego_i)
    fig.savefig(figname)
    plt.close()


def unit_test_sd_calc(veh1_s_matrix, veh2_s_array, veh1_d_matrix, veh2_d_array, veh1_v_matrix, ego_veh, veh, cost_func, mask, normed_distance_list):
    # Wird aufgerufen in safety_distance.sd_cost_for_ego
    street = scenario_acceleration_lane_street.create_road_with_acceleration_lane(s_length=200., number_of_lanes=1,
                                                                                  lane_width=3.5, s_change_start=40.,
                                                                                  s_change_end=80., v_max=50. / 3.6)
    veh1_s_matrix_m = veh1_s_matrix[mask]
    veh2_s_array_m = veh2_s_array[mask]
    veh1_d_matrix_m = veh1_d_matrix[mask]
    veh2_d_array_m = veh2_d_array[mask]
    veh1_v_matrix_m = veh1_v_matrix[mask]

    ego_i = counter()

    for i in range(len(veh1_s_matrix_m)):
        fig, ax = plt.subplots(figsize=(20, 10))
        add_street_to_plot(ax, street, 35., 110.)

        x_ego = veh1_s_matrix_m[i]
        y_ego = veh1_d_matrix_m[i]
        ego_v = veh1_v_matrix_m[i]

        veh_x = veh2_s_array_m[i]
        veh_y = veh2_d_array_m[i]

        heading_angle = 0

        add_vehicle_to_plot(ax, ego_veh.properties.length, ego_veh.properties.width,
                            ego_veh.properties.length_to_front_bumper, x_ego, y_ego, heading_angle, 'red')

        add_vehicle_to_plot(ax, veh.properties.length, veh.properties.width,
                            veh.properties.length_to_front_bumper, veh_x, veh_y, heading_angle, 'blue')

        alpha_list = np.arange(0, np.pi / 2 + 0.001, 0.001)
        distance_list = sd.safety_distance_envelop(ego_v, alpha_list)

        x = x_ego + np.sin(alpha_list) * distance_list
        y = y_ego + np.cos(alpha_list) * distance_list
        x_add = np.flip(x, axis=0)
        y_add = ((np.flip(y, axis=0) -y_ego) * -1) +y_ego
        x = np.append(x, x_add)
        y = np.append(y, y_add)

        ax.plot(x, y, color='red')
        cost = cost_func.calc_sd_costs2(normed_distance_list[i])

        title = 'Noramlized Safety Distance = ' + str(normed_distance_list[i]) + ',     Saftey Distance Cost = ' + str(cost)

        plt.title(title)

        figname = 'Unit_tests/Safety_distance/' + str(ego_i) + '_' + str(i)
        fig.savefig(figname)
        plt.close()


def plot_scenario(vehicle_list, sampling_range_t, t_list, street, delta_t=0.01, output_directory=''):

    x = []
    y = []
    heading_angle = []
    color_list = []
    width_list =[]
    lenght_list = []
    length_to_front_b = []
    t_list_delta_t = np.empty((len(t_list)-1)*int(sampling_range_t/delta_t))
    dt_in_sr_t = int(sampling_range_t/delta_t)
    for i in range(len(t_list)-1):
        for j in range(dt_in_sr_t):
            t_list_delta_t[i*dt_in_sr_t+j] = t_list[i] + delta_t * j
    for i in range(len(vehicle_list)):
        x_list, y_list, heading_angle_list = create_vehicle_plot_lists(vehicle_list[i], sampling_range_t, t_list, delta_t, street)
        x.append(x_list)
        y.append(y_list)
        heading_angle.append(heading_angle_list)
        width_list.append(vehicle_list[i].properties.width)
        lenght_list.append(vehicle_list[i].properties.length)
        length_to_front_b.append(vehicle_list[i].properties.length_to_front_bumper)
        if vehicle_list[i].typ == 0:
            color_list.append('red')
        elif vehicle_list[i].typ == 1:
            color_list.append('orange')
        elif vehicle_list[i].typ == 2:
            color_list.append('blue')
        else:
            color_list.append('green')

    for i in range(len(t_list_delta_t)):
        fig, ax = plt.subplots(figsize=(10, 3))
        add_street_to_plot(ax, street, 100., 210.)
        for j in range(len(vehicle_list)):
            add_vehicle_to_plot(ax, lenght_list[j], width_list[j], length_to_front_b[j], x[j][i], y[j][i], heading_angle[j][i], color_list[j])

        plt.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)

        red_patch = mpatches.Patch(color='red', label='Ego Vehicle')
        orange_patch = mpatches.Patch(color='orange', label='Involved')
        green_patch = mpatches.Patch(color='green', label='Unaffected')
        blue_patch = mpatches.Patch(color='blue', label='Affected')
        ax.legend(handles=[red_patch, orange_patch, blue_patch, green_patch], bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=4, mode="expand", borderaxespad=0.)
        # ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.502), loc=3, ncol=5, mode="expand", borderaxespad=0.)

        figname = output_directory + str(i)
        fig.savefig(figname, qualtiy=10)
        plt.close()


def create_vehicle_plot_lists(vehicle, sampling_range_t, t_list, delta_t, street):

    x_list = np.empty((len(t_list)-1)*int(sampling_range_t/delta_t))
    y_list = np.empty_like(x_list)
    heading_angle_list = np.empty_like(x_list)

    for i in range(len(t_list)-1):
        delta_s_reference = vehicle.trajectory.s_reference_list[i+1] - vehicle.trajectory.s_reference_list[i]
        delta_s_track = vehicle.trajectory.v_list[i] * sampling_range_t + 0.5 * vehicle.trajectory.at_list[
            i] * sampling_range_t * sampling_range_t

        s_r = vehicle.trajectory.s_reference_list[i]
        v = vehicle.trajectory.v_list[i]
        a = vehicle.trajectory.at_list[i]

        frames = int(math.ceil(sampling_range_t / delta_t))
        for j in range(frames):
            s_r += g_func.calc_ds(delta_t, v, a) * (delta_s_reference/delta_s_track)
            v += a*delta_t
            x, y = g_func.frenet_to_world_coordinates(street.reference_curve[0], street.reference_curve[1], s_r, vehicle.trajectory.path(s_r))
            heading_angle_street = g_func.calc_heading_angle(street.reference_curve[0], street.reference_curve[1], s_r)
            path_d = (vehicle.trajectory.path(s_r + 0.1) - vehicle.trajectory.path(s_r - 0.1)) / (2 * 0.1)
            heading_angle_path = np.arctan(path_d)
            heading_angle = heading_angle_path + heading_angle_street

            x_list[i*frames+j] = x
            y_list[i*frames + j] = y
            heading_angle_list[i*frames + j] = heading_angle

    return x_list, y_list, heading_angle_list


def plot_trajectory_set(trajectorie_set, sampling_range_t):

    plt.figure()

    ax_s = plt.subplot2grid((13, 1), (0, 0), rowspan=4)
    ax_d = plt.subplot2grid((13, 1), (5, 0), rowspan=2, sharex=ax_s)
    ax_v = plt.subplot2grid((13, 1), (8, 0), rowspan=2, sharex=ax_s)
    ax_a = plt.subplot2grid((13, 1), (11, 0), rowspan=2, sharex=ax_s)

    t_list = np.empty(np.size(trajectorie_set.v, axis=1))
    for i in range(len(t_list)):
        t_list[i] = i * sampling_range_t
    for i in range(np.size(trajectorie_set.v, axis=0)):
        ax_s.plot(t_list, trajectorie_set.s_ref[i, :])
        ax_d.plot(t_list, trajectorie_set.d[i, :])
        ax_v.plot(t_list, trajectorie_set.v[i, :])
        ax_a.plot(t_list, trajectorie_set.a[i, :])

    plt.xlabel('t [s]')
    ax_s.set_ylabel('s [m]')
    ax_d.set_ylabel('d [m]')
    ax_v.set_ylabel('v [m/s]')
    ax_a.set_ylabel('a [m/s^2]')

    # figname = '/home/diestmann/Documents/post_pro/Replan/' + str(int(self.t))
    plt.show()


if __name__ == "__main__":
    img = []
    for i in range(0, 1446):
        img.append(cv2.imread('/home/diestmann/Documents/post_pro/pyplot2/' + str(i) + '.png'))

    #display = cv2.namedWindow("display")
    #cv2.imshow(display, img[0])


    height, width, layers = img[1].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('/home/diestmann/Documents/post_pro/pyplot2/video.avi', fourcc, 100, (width, height))

    for j in range(0, 1446):
        video.write(img[j])

    video.release()
    cv2.destroyAllWindows()
