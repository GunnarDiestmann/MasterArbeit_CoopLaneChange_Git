import numpy as np
import matplotlib.pyplot as plt
import inspect

print("Achtung, float(delta_value > 0) durch (delta_value > 0) ersetzt!!")

class EvaluationFunctionComfortPos(object):
    def __init__(self, optimum, margin, treshold):
        self.optimum = optimum
        self.a_plus = treshold / (margin * margin)

    def cost(self, value):
        delta_value = value - self.optimum
        return self.a_plus * delta_value**2 * (delta_value > 0)


class EvaluationFunctionComfortNeg(object):
    def __init__(self, optimum, margin, treshold):
        self.optimum = optimum
        self.a_minus = treshold / (margin * margin)

    def cost(self, value):
        delta_value = value - self.optimum
        return self.a_minus * delta_value**2 * (delta_value < 0)


class EvaluationFunctionDiscomfortPos(object):
    def __init__(self, optimum, comf_margin, discomf_margin, treshold):
        self.comf_margin = comf_margin
        self.optimum = optimum
        self.b_plus = treshold / ((discomf_margin - comf_margin) ** 2.)

    def cost(self, value):
        delta_value = value - (self.optimum + self.comf_margin)
        return self.b_plus * delta_value**2 * (delta_value > 0)


class EvaluationFunctionDiscomfortNeg(object):
    def __init__(self, optimum, comf_margin, discomf_margin, treshold):
        self.comf_margin = comf_margin
        self.optimum = optimum
        self.b_minus = treshold / ((discomf_margin - comf_margin) ** 2.)

    def cost(self, value):
        delta_value = value - (self.optimum - self.comf_margin)
        return self.b_minus * delta_value**2 * (delta_value < 0)


class EvaluationFunctionInfeasPos(object):
    def __init__(self, optimum, inf_value, inf_margin, treshold):
        self.optimum = optimum
        self.inf_value = inf_value
        self.inf_margin = inf_margin
        self.c_plus = treshold / ((inf_margin * inf_margin) * np.exp(inf_margin))

    def cost(self, value):
        delta_value = value - (self.inf_value - self.inf_margin)
        cost = self.c_plus * delta_value**2 * np.exp(delta_value) * (delta_value > 0)
        # if cost > 10000.0:
        #     print "cost for value " + str(value) + " (delta_value =" + str(delta_value) + ") are " + str(cost)
        #     curframe = inspect.currentframe()
        #     calframe = inspect.getouterframes(curframe, 2)
        #     print 'caller name:', calframe[1][4][1]
        #     print " "
        return cost


class EvaluationFunctionInfeasNeg(object):
    def __init__(self, optimum, inf_value, inf_margin, treshold):
        self.optimum = optimum
        self.inf_value = inf_value
        self.inf_margin = inf_margin
        self.c_minus = treshold / ((inf_margin * inf_margin) * np.exp(inf_margin))

    def cost(self, value):
        delta_value = value - (self.inf_value + self.inf_margin)
        cost = self.c_minus * delta_value**2 * np.exp(abs(delta_value)) * (delta_value < 0)
        # if cost > 10000.0:
        #     print "cost for value " + str(value) + " (delta_value =" + str(delta_value) + ") are " + str(cost)
        #     curframe = inspect.currentframe()
        #     calframe = inspect.getouterframes(curframe, 2)
        #     print 'caller name:', calframe[1][4][1]
        #     print " "
        return cost

