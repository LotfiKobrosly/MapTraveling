import numpy as np

from utils.map_utils import cell_selector, get_intermediary_passage_points
from utils.constants import EXPLORATION_CONSTANT, RANDOM_STATE


def compute_uct(position, new_position, states_values):
    return states_values[tuple(new_position)]["mean_score"] + EXPLORATION_CONSTANT * np.sqrt(np.log(states_values[tuple(new_position)]["n_visits"]) / states_values[tuple(position)]["n_visits"])


def random_simulation(position, current_map):
    height, width = current_map.shape
    next_cell_found = False
    while not next_cell_found:
        angle = RANDOM_STATE.uniform(0, 1)
        new_cell = cell_selector(position, angle * 2 * np.pi)
        if (
            (new_cell[0] >= 0)
            and (new_cell[0] < height)
            and (new_cell[1] >= 0)
            and (new_cell[1] < width)
            and (current_map[new_cell[0], new_cell[1]] != 1)
        ):
            next_cell_found = True
            intermediary_passage_points = get_intermediary_passage_points(
                position, new_cell, current_map
            )
            for point in intermediary_passage_points:
                if current_map[point[0], point[1]] == 1:
                    next_cell_found = False
                    break
            if not next_cell_found:
                continue
            break
    return angle, new_cell


def backpropagation(trajectory, states_values, score, iteration_number):
    for position in trajectory:
        if not states_values.get(tuple(position), None) is None:
            states_values[tuple(position)]["cumulative_score"] += np.exp(-score)
            states_values[tuple(position)]["mean_score"] = states_values[tuple(position)]["cumulative_score"] / iteration_number
