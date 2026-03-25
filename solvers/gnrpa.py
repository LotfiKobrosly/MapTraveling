import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code, compute_heuristic_value
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    RANDOM_STATE,
    EPSILON,
    TAU,
    N_SAMPLES_TO_CHOOSE_FROM,
)


def gnrpa_step(
    position,
    goal,
    current_map,
    policy,
    sampling_radius=0.001,
    relevance_radius_list=[RELEVANCE_RADIUS],
) -> float:
    movement_list = list()
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

        for _ in range(N_SAMPLES_TO_CHOOSE_FROM):

            new_cell = [-1, -1]
            widening = 0.01  # to prevent the search of being stuck
            gaussian_filter = GaussianKernel([mean], sigma=sampling_radius)
            while not cell_is_reachable(position, new_cell, current_map):
                move = RANDOM_STATE.normal(mean, sampling_radius + widening, size=2)
                gaussian_filter = GaussianKernel(mean, sigma=sampling_radius + widening)
                widening += 0.01
                new_cell = continuous_cell_selector(position, move)
            movement_list.append(move)
        weights = np.array([gaussian_filter.pdf(move) for angle in movement_list])

    else:

        for _ in range(N_SAMPLES_TO_CHOOSE_FROM):
            new_cell = [-1, -1]
            widening = 0.01  # to prevent the search of being stuck
            while not cell_is_reachable(position, new_cell, current_map):
                move = RANDOM_STATE.uniform(-1, 1, size=2)
                new_cell = continuous_cell_selector(position, move)
            movement_list.append(move)

        weights = np.ones(len(movement_list)) / len(movement_list)
    assert not np.isnan(weights).any(), "NaN value found in weights computing"
    biases_values = np.array(
        [compute_heuristic_value(position, goal, move) for move in movement_list]
    )
    assert not np.isnan(biases_values).any(), "NaN value found in biases computing"
    biases_values = biases_values / np.sum(np.absolute(biases_values))
    probabilities = np.exp(weights / TAU + biases_values * 10)
    assert not np.isnan(
        probabilities
    ).any(), "NaN value found in probabilities computing"
    probabilities /= np.sum(probabilities)
    return movement_list[
        np.random.choice(
            list(range(len(movement_list))), size=1, p=probabilities, replace=False
        )[0]
    ]


def adapt_policy_gnrpa(
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
