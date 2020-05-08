import path_planing
import numpy as np
# import lanelet2


class StreetInfoROS(object):
    def __init__(self, center_line, init_lane_func, target_lane_func, v_max, s_lc_min, s_lc_max):
        self.center_line = center_line
        self.init_lane = init_lane_func
        self.target_lane = target_lane_func
        self.v_max = v_max
        self.s_lc_min = s_lc_min
        self.s_lc_max = s_lc_max

        self.init_lane_path = path_planing.Path(init_lane_func, self)
        self.target_lane_path = path_planing.Path(target_lane_func, self)

    def get_center_line_curvature(self, s):
        # This is a fake value under the presumption that the reference curve is almost straight
        # Curvature calculation could be added
        if np.isscalar(s):
            return 0
        else:
            return np.zeros_like(s)

    def get_center_line_yaw(self, s_ref):
        if np.isscalar(s_ref):
            direction_vec = lanelet2.geometry.interpolatedPointAtDistance(self.center_line, s_ref + 0.1) - lanelet2.geometry.interpolatedPointAtDistance(self.center_line, s_ref)
            yaw = np.arctan2(direction_vec.y, direction_vec.x)
            return yaw
        else:
            yaw = np.empty_like(s_ref)
            for i in range(len(s_ref)):
                direction_vec = lanelet2.geometry.interpolatedPointAtDistance(self.center_line, s_ref[i] + 0.1) - lanelet2.geometry.interpolatedPointAtDistance(self.center_line, s_ref[i])
                yaw[i] = np.arctan2(direction_vec.y, direction_vec.x)
            return yaw

    def get_xy_from_sd(self, s, d):
        xy_pos_ref = lanelet2.geometry.interpolatedPointAtDistance(self.center_line, s)
        tangent = xy_pos_ref - lanelet2.geometry.interpolatedPointAtDistance(self.center_line, s - 0.1)
        normed_tangent = tangent / np.sqrt(tangent.x ** 2 + tangent.y ** 2)
        normal = lanelet2.core.BasicPoint2d(abs(normed_tangent.y), normed_tangent.x)
        xy_curr = lanelet2.core.BasicPoint2d(xy_pos_ref.x + d * normal.x, xy_pos_ref.y + d * normal.y)

        return xy_curr.x, xy_curr.y


class LaneChangeStreetInfo_fake(object):
    def __init__(self, center_line, init_lane, target_lane, v_max, s_lc_min, s_lc_max):
        self.center_line = center_line
        self.init_lane = init_lane
        self.target_lane = target_lane
        self.v_max = v_max
        self.s_lc_min = s_lc_min
        self.s_lc_max = s_lc_max

        self.init_lane_path = path_planing.Path(init_lane, self)
        self.target_lane_path = path_planing.Path(target_lane, self)

    def get_center_line_curvature(self, s):
        return 0.

    def get_center_line_yaw(self, s_ref, d_ref):
        return 0.

    def get_xy_from_sd(self, s, d):
        return s, d


class ReferenceCurve(object):
    def __init__(self, x_ref, y_ref):
        self.x_ref = x_ref
        self.y_ref = y_ref


class LaneMarker(object):
    """
    This class represents a single lanemarker of a road
    """
    def __init__(self, func, typ, s_start, s_end):
        """
        lane marker init

        :type func: function
        :param func: function that describes the shape of the lane marker
        :type typ: int
        :param typ: 0: solid lane, 1: dashed lane
        :type s_start: float
        :param s_start: gives s of reference road where the lane marker starts
        :type s_end: float
        :param s_end: gives s of reference road where the lane marker ends
        """

        self.func = func
        self.typ = typ
        self.s_start = s_start
        self.s_end = s_end

    def value(self, s):
        """
        returns orthogonal distance d for reference curve for a given s (look at frenet coordinates)
        :param s: s of reference curve (x(s), y(s))
        :return: d(s)
        """

        return self.func(s)


class Lane(object):
    """
    This class represents a single lane
    """

    def __init__(self, func, index, width, s_start, s_end, v_max=None):
        """
        lane init

        :type func: function
        :param func: function that describes the path of a lane
        :type index: int
        :param index: index of the lane
        :type width: float
        :param width: width of the lane
        :type s_start: float
        :param s_start: when does the lane start (given in s of reference curve)
        :type s_end: float
        :param s_end: when does the lane end (given in s of reference curve)
        :type v_max: float
        :param v_max: speed limit on this lane. Just needs to be given when it is different to road speed limit
        """
        self.func = func
        self.index = index
        self.width = width
        self.s_start = s_start
        self.s_end = s_end
        self.v_max = v_max

    def value(self, s):
        """
        returns orthogonal distance d for reference curve for a given s (look at frenet coordinates)
        :param s: s of reference curve (x(s), y(s))
        :return: d(s)
        """
        return self.func(s)


class Street(object):

    """
    This class represents a street in certain segment
    """
    def __init__(self, reference_curve, lane_markers, lanes, s_length, v_max, x_start=0., y_start=0., alpha_start=0.):

        """
        adds reference curve, lanes, and lane markers to the street

        :type reference_curve: list[function]
        :param reference_curve: list[0] = x[s], list[1] = y[s]
        :type lane_markers: list[LaneMarker]
        :param lane_markers: list with the lane marker functions, in Frenet coordinates -> d(s_reference)
        :type lanes: list[Lane]
        :param lanes: list with the functions of lane middle, in Frenet coordinates -> d(s_reference)
        :type s_length: float
        :param s_length: length of the road segment (arc length of reference  curve)
        :type v_max: float
        :param v_max: max allowed velocity
        :param x_start: road segment starting at x_start
        :param y_start: road segment starting at y_start
        :param alpha_start: road start at an angle alpha start in world coordinates
        """

        self.reference_curve = reference_curve
        self.lane_markers = lane_markers
        self.lanes = lanes
        self.s_length = s_length
        self.v_max = v_max
        self.start = [x_start, y_start, alpha_start]


class StreetWithAccelerationLane(Street):
    """
    This class is a subclass of the class street. It has additional information for a street with acceleration lane
    """
    def __init__(self, reference_curve, lane_markers, lanes, s_length, v_max, s_change_start, s_change_end,
                 x_start=0., y_start=0., alpha_start=0.):
        """
        adds reference curve, lanes, and lane markers to the street

        :type reference_curve: list
        :param reference_curve: list[0] = x[s], list[1] = y[s]
        :type lane_markers: list
        :param lane_markers: list with the lane marker functions, in Frenet coordinates -> d(s_reference)
        :type lanes: list[Lane]
        :param lanes: list with the functions of lane middle, in Frenet coordinates -> d(s_reference)
        :type s_length: float
        :param s_length: length of the road segment (arc length of reference  curve)
        :type v_max: float
        :param v_max: max allowed velocity
        :type s_change_start: float
        :param s_change_start: gives s of reference curve where the lane changer from the acceleration lane can start
        :type s_change_end: float
        :param s_change_end: give last possible s for the lane chang from acceleration lane to the street
        :param x_start: road segment starting at x_start
        :param y_start: road segment starting at y_start
        :param alpha_start: road start at an angle alpha start in world coordinates
        """

        Street.__init__(self, reference_curve, lane_markers, lanes, s_length, v_max, x_start, y_start, alpha_start)
        self.s_change_start = s_change_start
        self.s_change_end = s_change_end
