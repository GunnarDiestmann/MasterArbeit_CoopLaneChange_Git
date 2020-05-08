import Street
import Polynomial_funktion
import numpy as np


class StraightLaneFunction(object):
    """
    This class gives a function for x(s) when a the reference curve is a straight line. In that case x=s
    """
    def value(self, s):
        return s


class LineParallelToXAxis(object):
    """
    This class describes a line with a certain distance to a reference line
    """
    def __init__(self, distance):
        self.a = distance

    def value(self, x):
        return self.a + x*0


class MergeLaneMarkerFunction(object):
    """
    This class describes a function for a lane marker that is merging into another one
    """
    def __init__(self, merge_start, merge_length, d_start, d_merge):
        """
        :type merge_start: float
        :param merge_start: position (s) where the lane starts to merge
        :type merge_length: float
        :param merge_length: length in which the lane merging will happen
        :type d_start: float
        :param d_start: distance from the reference curve before the merging starts
        :type d_merge: float
        :param d_merge: distance from the reference curve of the lane marker the original lane marker merges into
        """
        self.merge_start = merge_start
        self.merge_end = merge_start + merge_length
        self.d_start = d_start
        self.d_merge = d_merge

    def value(self, x):
        if x < self.merge_start:
            return self.d_start + 0 * x
        else:
            merge_poly = Polynomial_funktion.Poly5(self.merge_start, self.d_start, self.merge_end, self.d_merge, 0., 0.,
                                                   0., 0.)
            return merge_poly.value(x)


def create_reference_curve_for_straight_road():
    """
    This function creates the reference curve x(s), y(s) when the road is a straight road
    :return: [x(s), y(s)]
    """
    reference_curve_x = StraightLaneFunction()
    reference_curve_y = LineParallelToXAxis(distance=0.)
    return [reference_curve_x.value, reference_curve_y.value]


def create_lanes_road_with_acceleration_lane(number_of_lanes, lane_width, s_start, s_end, s_change_end):
    """
    This function creates a list of lanes with the same lane width and one extra for the acceleration lane.
    It is just for streets where the lanes keep the same distance to the reference curve and don't vary the width
    :type number_of_lanes: int
    :param number_of_lanes: number of lanes, the acceleration lane not included
    :type lane_width: float
    :param lane_width: width of a single lane
    :type s_start: float
    :param s_start: start of the road
    :type s_end: float
    :param s_end: end of the road
    :type s_change_end: float
    :param s_change_end: when does the acceleration lane end
    :return: list with all lanes of the street
    """
    # Create a function for every lane
    lane_functions_list = []
    for i in range(number_of_lanes + 1):
        distance_form_reference_curve = -(lane_width/2) + i*lane_width
        lane_functions_list.append(LineParallelToXAxis(distance_form_reference_curve))

    # Create a list with all lanes in it
    lane_list = []
    lane_list.append(Street.Lane(lane_functions_list[0].value, 0, lane_width, s_start, s_change_end))
    for i in range(len(lane_functions_list)-1):
        lane_list.append(Street.Lane(lane_functions_list[i+1].value, i+1, lane_width, s_start, s_end))

    return lane_list


def create_lane_markers_road_with_acceleration_lane(number_of_lanes, lane_width, s_start, s_end, s_change_start,
                                                    s_change_end):
    """
    Creates a list of lane markers for a road with acceleration lane
    It is just for streets where the lanes keep the same distance to the reference curve and don't vary the width
    :type number_of_lanes: int
    :param number_of_lanes: number of lanes, the acceleration lane not included
    :type lane_width: float
    :param lane_width: width of a single lane
    :type s_start: float
    :param s_start: start of the road
    :type s_end: float
    :param s_end: end of the road
    :type s_change_start: float
    :param s_change_start: gives the position after which vehicles can switch lanes form acceleration lane to road
    :type s_change_end: float
    :param s_change_end: when does the acceleration lane end
    :return:
    """

    # A list with all functions of the lane markers is created
    lane_marker_functions = []
    merge_length = 20.
    # Function for the acceleration lane marker that merges into the very right road lane marker is added
    lane_marker_functions.append(MergeLaneMarkerFunction(s_change_end, merge_length, -lane_width, 0.))

    for i in range(number_of_lanes + 1):
        lane_marker_functions.append(LineParallelToXAxis(i*lane_width))

    # Given the functions now a list of type lane marker can be created
    lane_marker_list = []
    # Add acceleration lane marker right
    lane_marker_list.append(Street.LaneMarker(lane_marker_functions[0].value, 0, s_start, s_change_end+merge_length))
    # Add acceleration lane marker left/road lane marker right
    lane_marker_list.append(Street.LaneMarker(lane_marker_functions[1].value, 0, s_start, s_change_start))
    lane_marker_list.append(Street.LaneMarker(lane_marker_functions[1].value, 1, s_change_start, s_change_end +
                                              merge_length))
    lane_marker_list.append(Street.LaneMarker(lane_marker_functions[1].value, 0, s_change_end + merge_length, s_end))
    # Add dashed lane markers in the middle of the road
    if len(lane_marker_functions) > 3:
        dashed_lanes = np.arange(2, len(lane_marker_functions)-1, 1)
        for i in dashed_lanes:
            lane_marker_list.append(Street.LaneMarker(lane_marker_functions[i].value, 1, s_start, s_end))

    # Add solid lane marker on the very right of the road
    lane_marker_list.append(Street.LaneMarker(lane_marker_functions[-1].value, 0, s_start, s_end))

    return lane_marker_list


def create_road_with_acceleration_lane(s_length, number_of_lanes, lane_width, s_change_start, s_change_end, v_max):
    """
    This function will create a street with an acceleration lane. The lane_width of all lanes will be the same
    The acceleration lane will have index 0. The other lanes of the road will have a increasing index from the most left
    ont to the most right one, starting at 1
    :type s_length: float
    :param s_length: length of the road
    :type number_of_lanes: int
    :param number_of_lanes: number of lanes, the acceleration lane not included
    :type lane_width: float
    :param lane_width: width of a single lane
    :type s_change_start: float
    :param s_change_start: gives the position after which vehicles can switch lanes form acceleration lane to road
    :type s_change_end: float
    :type v_max: float
    :param v_max: speed limit given for that road
    :return:
    """
    s_start = 0.
    s_end = s_start + s_length

    reference_curve = create_reference_curve_for_straight_road()

    lane_list = create_lanes_road_with_acceleration_lane(number_of_lanes, lane_width, s_start, s_end, s_change_end)

    lane_marker_list = create_lane_markers_road_with_acceleration_lane(number_of_lanes, lane_width, s_start, s_end,
                                                                       s_change_start, s_change_end)

    street = Street.StreetWithAccelerationLane(reference_curve, lane_marker_list, lane_list, s_length, v_max,
                                               s_change_start, s_change_end)
    return street
