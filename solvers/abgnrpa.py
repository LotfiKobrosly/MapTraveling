import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code_action, code_position
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    N_GMM_COMPONENTS,
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
    if len(policy) > 1:

        if position in policy.keys():
            mean = policy[position]
        else:
            coefficients = np.zeros((len(policy)))
            angles = np.zeros((len(policy)))
            radius_index = 0
            while np.sum(coefficients) <= EPSILON and radius_index < len(relevance_radius_list):
                gaussian_filter = GaussianKernel(position, sigma=relevance_radius_list[radius_index])
                for counter, (key, value) in enumerate(policy.items()):
                    coefficients[counter] = gaussian_filter.pdf(key)
                    angles[counter] = value
                radius_index += 1
            coefficients /= np.sum(coefficients)
            mean = angles @ coefficients.T


        normalized_angles = list()
        for _ in range(N_SAMPLES_TO_CHOOSE_FROM):

            new_cell = [-1, -1]
            widening = 0.01 # to prevent the search from being stuck
            gaussian_filter = GaussianKernel([mean], sigma=sampling_radius)
            while not cell_is_reachable(new_cell, current_map):
                normalized_angle = RANDOM_STATE.normal(mean, sampling_radius + widening, size=1)[0]
                gaussian_filter = GaussianKernel([mean], sigma=sampling_radius + widening)
                widening += 0.01
                new_cell = continuous_cell_selector(position, normalized_angle)
            normalized_angles.append(normalized_angle)
        weights = np.array(
            [
                gaussian_filter.pdf(angle)
                for angle in normalized_angles
            ]
        )

    else:
        normalized_angles = list()
        for _ in range(N_SAMPLES_TO_CHOOSE_FROM):
            new_cell = [-1, -1]
            widening = 0.01 # to prevent the search of being stuck
            while not cell_is_reachable(new_cell, current_map):
                normalized_angle = RANDOM_STATE.uniform(
                    0, 1, size=1
                )[0]
                widening += 0.01
                new_cell = continuous_cell_selector(position, normalized_angle)
            normalized_angles.append(normalized_angle)
            
        weights = np.ones(len(normalized_angles)) / len(normalized_angles)
    biases_values = np.array(
        [
            heuristic_values.get(position, goal, angle)
            for angle in normalized_angles
        ]
    )
    biases_values = biases_values / np.sum(np.absolute(biases_values))
    probabilities = np.exp(weights / TAU + biases_values * 10)
    probabilities /= np.sum(probabilities)
    normalized_angle = np.random.choice(
        normalized_angles, size=1, p=probabilities, replace=False
    )[0]
    for angle in normalized_angles:
        previous_value = heuristic_values.get(position, goal, angle)
        sign = -1 / len(normalized_angles)
        if angle == normalized_angle:
            sign = 1
        heuristic_values.set(position, angle, previous_value + sign * GAMMA * score)

    return normalized_angle
