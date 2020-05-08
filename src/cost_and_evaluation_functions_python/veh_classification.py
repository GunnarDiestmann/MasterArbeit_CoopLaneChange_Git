import numpy as np
import post_pro


class Classification(object):
    involved_veh_keys = None
    influenced_veh_keys = None
    unaffected_veh_key = None

    def __init__(self):
        self.involved_veh_keys = []
        self.influenced_veh_keys = []
        self.unaffected_veh_key = []

    def classify_for_lane_change(self, s_matrix, v_matrix, vehicle_list, key_t_interact):
        for i in range(1, len(vehicle_list.veh_list)):
            self.unaffected_veh_key.append(i)

        key = 1
        max_key = vehicle_list.num_of_veh_tl
        while s_matrix[key][key_t_interact] > s_matrix[0][key_t_interact] and key < max_key:
            key += 1

        if vehicle_list.num_of_veh_tl + 1 > key > 1:
            # determine if preceding veh interacts
            if self.determine_interaction_lane_change(vehicle_list.veh_list[0], vehicle_list.veh_list[key-1], s_matrix[0][key_t_interact],
                                                      v_matrix[0][key_t_interact], s_matrix[key-1][key_t_interact]):
                self.involved_veh_keys.append(key-1)
                self.unaffected_veh_key.remove(key-1)

        if key > 0:
            # Determine if following veh interacts
            if self.determine_interaction_lane_change(vehicle_list.veh_list[key], vehicle_list.veh_list[0], s_matrix[key][key_t_interact],
                                                      v_matrix[key][key_t_interact], s_matrix[0][key_t_interact]):
                self.involved_veh_keys.append(key)
                self.unaffected_veh_key.remove(key)
            else:
                self.influenced_veh_keys.append(key)
                self.unaffected_veh_key.remove(key)

            for i in range(key+1, max_key+1):
                self.influenced_veh_keys.append(i)
                self.unaffected_veh_key.remove(i)

    def determine_interaction_lane_change(self, veh, veh_prec, veh_s, veh_v, prec_veh_s):
        distance = (prec_veh_s - veh_prec.properties.length_to_rear_bumper) - \
                   (veh_s + veh.properties.length_to_front_bumper)
        sd_required = 0.5 * veh_v * 3.6

        if sd_required > distance:
            return True
        else:
            return False
