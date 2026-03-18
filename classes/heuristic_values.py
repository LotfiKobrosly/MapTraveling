from utils.basic_functions import *


class HeuristicValues(object):

    def __init__(self, bias_factor: float):
        self.bias_factor = bias_factor
        self.values = dict()

    def get_key(self, position, move):
        return code(position) + code(move)

    def get(self, position, goal, move):
        key = self.get_key(position, move)
        if self.values.get(key, None) is None:
            self.values[key] = self.bias_factor * compute_heuristic_value(
                position, goal, move
            )
        return self.values[key]

    def set(self, position, move, value):
        key = self.get_key(position, move)
        self.values[key] = self.bias_factor * value
