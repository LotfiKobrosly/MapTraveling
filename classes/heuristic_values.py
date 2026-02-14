from utils.basic_functions import *

class HeuristicValues(object):

    def __init__(self, bias_factor: float):
        self.bias_factor = bias_factor
        self.values = dict()

    def get_key(self, position, angle):
        return (position[0], position[1], code_action(angle))

    def get(self, position, goal, angle):
        key = self.get_key(position, angle)
        if self.values.get(key, None) is None:
            self.values[key] = self.bias_factor * compute_heuristic_value(position, goal, angle)
        return self.values[key]

    def set(self, position, angle, value):
        key = self.get_key(position, angle)
        self.values[key] = self.bias_factor * value