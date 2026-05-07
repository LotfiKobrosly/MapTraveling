import numpy as np

from scipy.ndimage import gaussian_filter

from utils.basic_functions import code
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_STATE,
    LEARNING_RATE,
)


def nrpa_step(
    position,
    current_map,
    policy,
    sampling_radius=0.001,
    relevance_radius_list=[RELEVANCE_RADIUS],
):
    if len(policy) > 1:

        if position in policy.keys():
            mean = np.array(policy[position])
        else:
            coefficients = np.zeros((len(policy)))
            movements = np.zeros((len(policy), 2))
            radius_index = 0
            while np.sum(coefficients) < 0.1 and radius_index < len(
                relevance_radius_list
            ):
                gaussian_filter = GaussianKernel(
                    position, sigma=relevance_radius_list[radius_index]
                )
                for counter, (key, value) in enumerate(policy.items()):
                    coefficients[counter] = gaussian_filter.pdf(key)
                    movements[counter] = value
                radius_index += 1
            assert not np.isnan(
                coefficients
            ).any(), "NaN value found in coefficients computing"
            coefficients /= np.sum(coefficients)
            mean = np.array(coefficients @ movements)
        new_cell = [-1, -1]
        widening = 0.01  # to prevent the search from being stuck
        while not cell_is_reachable(position, new_cell, current_map):
            move = RANDOM_STATE.normal(mean, sampling_radius + widening, size=2)
            widening += 0.01
            if int(widening / 0.01) % 100 == 0:
                print("Mean ", mean)
                print("Sigma: ", sampling_radius + widening)
                print("Reached ", int(widening / 0.01), " iterations of widening")
            new_cell = continuous_cell_selector(position, move)

    else:
        new_cell = [-1, -1]
        while not cell_is_reachable(position, new_cell, current_map):
            move = RANDOM_STATE.uniform(-1, 1, size=2)
            new_cell = continuous_cell_selector(position, move)
    return move, new_cell


def adapt_policy_nrpa(
    best_trajectory,
    policy,
    learning_rate,
):
    for point_index, point in enumerate(best_trajectory[:-1]):

        new_move = np.array(best_trajectory[-1]) - np.array(point)
        new_move /= np.linalg.norm(new_move)
        if code(point) in policy.keys():
            previous_move = np.array(policy[code(point)])
            policy[code(point)] = np.array(policy[code(point)]) + learning_rate * (
                new_move - previous_move
            )
        else:
            policy[code(point)] = new_move
        policy[code(point)] = code(
            policy[code(point)] / np.linalg.norm(np.array(policy[code(point)]))
        )
    return policy
