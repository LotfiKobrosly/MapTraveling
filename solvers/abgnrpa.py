import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    TWO_PI,
    RANDOM_STATE,
    EPSILON,
    TAU,
    N_SAMPLES_TO_CHOOSE_FROM,
    GAMMA,
)
from classes.heuristic_values import HeuristicValues


def abgnrpa_step(
    position,
    goal,
    current_map,
    policy,
    heuristic_values,
    score,
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
            while not cell_is_reachable(new_cell, current_map):
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
            while not cell_is_reachable(new_cell, current_map):
                move = RANDOM_STATE.uniform(-1, 1, size=2)
                new_cell = continuous_cell_selector(position, move)
            movement_list.append(move)

        weights = np.ones(len(movement_list)) / len(movement_list)
    assert not np.isnan(weights).any(), "NaN value found in weights computing"
    biases_values = np.array(
        [heuristic_values.get(position, goal, move) for move in movement_list]
    )
    assert not np.isnan(biases_values).any(), "NaN value found in biases computing"
    biases_values = biases_values / np.sum(np.absolute(biases_values))
    probabilities = np.exp(weights / TAU + biases_values * 10)
    assert not np.isnan(
        probabilities
    ).any(), "NaN value found in probabilities computing"
    # print(probabilities)
    # print(movement_list)
    probabilities /= np.sum(probabilities)
    chosen_move = movement_list[
        np.random.choice(
            list(range(len(movement_list))), size=1, p=probabilities, replace=False
        )[0]
    ]
    for move in movement_list:
        previous_value = heuristic_values.get(position, goal, move)
        sign = -1 / len(movement_list)
        if np.all(move == chosen_move):
            sign = 1
        heuristic_values.set(position, move, previous_value + sign * GAMMA * score)

    return chosen_move
