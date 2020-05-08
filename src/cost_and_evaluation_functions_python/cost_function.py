import numpy as np
import matplotlib.pyplot as plt

import evaluation_functions


class CostFunction(object):
    def __init__(self, delta_t):
        self.delta_t = delta_t

        self.comf_treshold = 10.
        self.discomf_treshold = 100.
        self.inf_treshold = 10000.

        self.v_opt = 50./3.6
        self.v_comf_margin_pos = 10./3.6
        self.v_comf_margin_neg = 50./3.6
        self.v_discomf_margin_pos = 15./3.6
        self.v_inf_value_neg = -0.5
        self.v_inf_margin_neg = 1.

        self.at_opt = 0.
        self.at_comf_margin_pos = 1.5
        self.at_comf_margin_neg = 1.5
        self.at_discomf_margin_pos = 3.5
        self.at_discomf_margin_neg = 5.
        self.at_inf_value_pos = 3.5
        self.at_inf_margin_pos = 0.5
        self.at_inf_value_neg = -8.
        self.at_inf_margin_neg = 0.5

        self.an_opt = 0.
        self.an_comf_margin_pos = 2.5
        self.an_discomf_margin_pos = 5.
        self.an_inf_value_pos = 8.0
        self.an_inf_margin_pos = 0.5

        self.czct_opt = 3.
        self.czct_comf_margin_neg = 1.
        self.czct_discomf_margin_neg = 2.5
        self.czct_inf_value_pos = 0.
        self.czct_inf_margin_pos = 0.1

        # parameters for safety distance. Value one is when the vehicle keeps the safety distance given by German law
        self.sd_opt = 1.1
        self.sd_comf_margin_neg = 0.1
        self.sd_discomf_margin_neg = 0.3
        self.sd_inf_value_pos = 0.
        self.sd_inf_margin_pos = 0.1

        # parameters cost factor function for safety distance during lane change
        self.sd_factor_start = 1.
        self.sd_factor_end = 10.
        self.sd_factor_time = 3.

        # v
        self.v_comf_cost_pos = evaluation_functions.EvaluationFunctionComfortPos(self.v_opt, self.v_comf_margin_pos, self.comf_treshold)
        self.v_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.v_opt, self.v_comf_margin_neg, self.comf_treshold)
        self.v_discomf_cost_pos = evaluation_functions.EvaluationFunctionDiscomfortPos(self.v_opt, self.v_comf_margin_pos, self.v_discomf_margin_pos, self.discomf_treshold)
        self.v_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.v_opt, self.v_inf_value_neg, self.v_inf_margin_neg, self.inf_treshold)

        # a tangential
        self.at_comf_cost_pos = evaluation_functions.EvaluationFunctionComfortPos(self.at_opt, self.at_comf_margin_pos, self.comf_treshold)
        self.at_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.at_opt, self.at_comf_margin_neg, self.comf_treshold)
        self.at_discomf_cost_pos = evaluation_functions.EvaluationFunctionDiscomfortPos(self.at_opt,
                                                                                       self.at_comf_margin_pos,
                                                                                       self.at_discomf_margin_pos,
                                                                                       self.discomf_treshold)
        self.at_discomf_cost_neg = evaluation_functions.EvaluationFunctionDiscomfortNeg(self.at_opt,
                                                                                       self.at_comf_margin_neg,
                                                                                       self.at_discomf_margin_neg,
                                                                                       self.discomf_treshold)
        self.at_inf_cost_pos = evaluation_functions.EvaluationFunctionInfeasPos(self.at_opt, self.at_inf_value_pos,
                                                                               self.at_inf_margin_pos, self.inf_treshold)
        self.at_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.at_opt, self.at_inf_value_neg,
                                                                               self.at_inf_margin_neg, self.inf_treshold)

        # a lateral / normal
        self.an_comf_cost_pos = evaluation_functions.EvaluationFunctionComfortPos(self.an_opt, self.an_comf_margin_pos,
                                                                                  self.comf_treshold)
        self.an_discomf_cost_pos = evaluation_functions.EvaluationFunctionDiscomfortPos(self.an_opt,
                                                                                        self.an_comf_margin_pos,
                                                                                        self.an_discomf_margin_pos,
                                                                                        self.discomf_treshold)
        self.an_inf_cost_pos = evaluation_functions.EvaluationFunctionInfeasPos(self.an_opt, self.an_inf_value_pos,
                                                                                self.an_inf_margin_pos,
                                                                                self.inf_treshold)

        # czct
        self.czct_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.czct_opt, self.czct_comf_margin_neg,
                                                                                  0.0)
        self.czct_discomf_cost_neg = evaluation_functions.EvaluationFunctionDiscomfortNeg(self.czct_opt,
                                                                                        self.czct_comf_margin_neg,
                                                                                        self.czct_discomf_margin_neg,
                                                                                        self.discomf_treshold)
        self.czct_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.czct_opt, self.czct_inf_value_pos,
                                                                                self.czct_inf_margin_pos,
                                                                                self.inf_treshold)

        # safety distance
        self.sd_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.sd_opt, self.sd_comf_margin_neg,
                                                                                  self.comf_treshold)
        self.sd_discomf_cost_neg = evaluation_functions.EvaluationFunctionDiscomfortNeg(self.sd_opt,
                                                                                        self.sd_comf_margin_neg,
                                                                                        self.sd_discomf_margin_neg,
                                                                                        self.discomf_treshold)
        self.sd_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.sd_opt, self.sd_inf_value_pos,
                                                                                self.sd_inf_margin_pos,
                                                                                self.inf_treshold)

    def reset_eval_functions(self):
        # v
        self.v_comf_cost_pos = evaluation_functions.EvaluationFunctionComfortPos(self.v_opt, self.v_comf_margin_pos,
                                                                                 self.comf_treshold)
        self.v_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.v_opt, self.v_comf_margin_neg,
                                                                                 self.comf_treshold)
        self.v_discomf_cost_pos = evaluation_functions.EvaluationFunctionDiscomfortPos(self.v_opt,
                                                                                       self.v_comf_margin_pos,
                                                                                       self.v_discomf_margin_pos,
                                                                                       self.discomf_treshold)
        self.v_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.v_opt, self.v_inf_value_neg,
                                                                               self.v_inf_margin_neg, self.inf_treshold)

        # a tangential
        self.at_comf_cost_pos = evaluation_functions.EvaluationFunctionComfortPos(self.at_opt, self.at_comf_margin_pos,
                                                                                  self.comf_treshold)
        self.at_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.at_opt, self.at_comf_margin_neg,
                                                                                  self.comf_treshold)
        self.at_discomf_cost_pos = evaluation_functions.EvaluationFunctionDiscomfortPos(self.at_opt,
                                                                                        self.at_comf_margin_pos,
                                                                                        self.at_discomf_margin_pos,
                                                                                        self.discomf_treshold)
        self.at_discomf_cost_neg = evaluation_functions.EvaluationFunctionDiscomfortNeg(self.at_opt,
                                                                                        self.at_comf_margin_neg,
                                                                                        self.at_discomf_margin_neg,
                                                                                        self.discomf_treshold)
        self.at_inf_cost_pos = evaluation_functions.EvaluationFunctionInfeasPos(self.at_opt, self.at_inf_value_pos,
                                                                                self.at_inf_margin_pos,
                                                                                self.inf_treshold)
        self.at_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.at_opt, self.at_inf_value_neg,
                                                                                self.at_inf_margin_neg,
                                                                                self.inf_treshold)

        # a lateral / normal
        self.an_comf_cost_pos = evaluation_functions.EvaluationFunctionComfortPos(self.an_opt, self.an_comf_margin_pos,
                                                                                  self.comf_treshold)
        self.an_discomf_cost_pos = evaluation_functions.EvaluationFunctionDiscomfortPos(self.an_opt,
                                                                                        self.an_comf_margin_pos,
                                                                                        self.an_discomf_margin_pos,
                                                                                        self.discomf_treshold)
        self.an_inf_cost_pos = evaluation_functions.EvaluationFunctionInfeasPos(self.an_opt, self.an_inf_value_pos,
                                                                                self.an_inf_margin_pos,
                                                                                self.inf_treshold)

        # czct
        self.czct_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.czct_opt,
                                                                                    self.czct_comf_margin_neg,
                                                                                    0.0)
        self.czct_discomf_cost_neg = evaluation_functions.EvaluationFunctionDiscomfortNeg(self.czct_opt,
                                                                                          self.czct_comf_margin_neg,
                                                                                          self.czct_discomf_margin_neg,
                                                                                          self.discomf_treshold)
        self.czct_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.czct_opt,
                                                                                  self.czct_inf_value_pos,
                                                                                  self.czct_inf_margin_pos,
                                                                                  self.inf_treshold)

        # safety distance
        self.sd_comf_cost_neg = evaluation_functions.EvaluationFunctionComfortNeg(self.sd_opt, self.sd_comf_margin_neg,
                                                                                  self.comf_treshold)
        self.sd_discomf_cost_neg = evaluation_functions.EvaluationFunctionDiscomfortNeg(self.sd_opt,
                                                                                        self.sd_comf_margin_neg,
                                                                                        self.sd_discomf_margin_neg,
                                                                                        self.discomf_treshold)
        self.sd_inf_cost_neg = evaluation_functions.EvaluationFunctionInfeasNeg(self.sd_opt, self.sd_inf_value_pos,
                                                                                self.sd_inf_margin_pos,
                                                                                self.inf_treshold)

    def calc_at_costs(self, at):
        """
        Calculate the costs for lateral acceleration
        :type at: numpy array or float
        :param at: acceleration values
        :return: at costs
        """
        at_cost = self.at_comf_cost_pos.cost(at) * self.delta_t + self.at_comf_cost_neg.cost(
            at) * self.delta_t + self.at_discomf_cost_pos.cost(at) * self.delta_t + self.at_discomf_cost_neg.cost(
            at) * self.delta_t + self.at_inf_cost_pos.cost(at) * self.delta_t + self.at_inf_cost_neg.cost(
            at) * self.delta_t
        return np.sum(at_cost)

    def calc_v_costs(self, v):
        """
        Calculate the costs for lateral acceleration
        :type v: numpy array or float
        :param v: velocity values
        :return: v_costs
        """
        v_cost = self.v_comf_cost_pos.cost(v) * self.delta_t + self.v_comf_cost_neg.cost(v) * self.delta_t + \
                 self.v_discomf_cost_pos.cost(v) * self.delta_t + self.v_inf_cost_neg.cost(v) * self.delta_t

        return np.sum(v_cost)

    def calc_sd_costs(self, normed_distance):
        """

        :param normed_distance:
        :return:
        """
        sd_costs = self.sd_comf_cost_neg.cost(normed_distance) * self.delta_t + self.sd_discomf_cost_neg.cost(
            normed_distance) * self.delta_t + self.sd_inf_cost_neg.cost(normed_distance) * self.delta_t

        return np.sum(sd_costs)

    def calc_sd_costs_for_mask(self, normed_distance):
        """

        :param normed_distance:
        :return:
        """
        sd_costs = self.sd_comf_cost_neg.cost(normed_distance) * self.delta_t + self.sd_discomf_cost_neg.cost(
            normed_distance) * self.delta_t + self.sd_inf_cost_neg.cost(normed_distance) * self.delta_t

        return sd_costs

    def calc_lc_costs(self, veh_state, following_veh_state, t_to_lc_end):
        d = veh_state.s_ref - following_veh_state.s_ref
        a_nece = (d + (veh_state.v - following_veh_state.v) * t_to_lc_end - 1.8 * following_veh_state.v) / (
                    1.8 * t_to_lc_end + 0.5 * t_to_lc_end * t_to_lc_end)

        a_nece = a_nece * (a_nece<0)

        lc_cost = self.at_comf_cost_pos.cost(a_nece) * t_to_lc_end + self.at_comf_cost_neg.cost(
            a_nece) * t_to_lc_end + self.at_discomf_cost_pos.cost(a_nece) * t_to_lc_end + self.at_discomf_cost_neg.cost(
            a_nece) * t_to_lc_end + self.at_inf_cost_pos.cost(a_nece) * t_to_lc_end + self.at_inf_cost_neg.cost(
            a_nece) * t_to_lc_end

        return lc_cost
