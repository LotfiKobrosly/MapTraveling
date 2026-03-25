import numpy as np

from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
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
                print("Sigma: ", sampling_radius + widening_factor)
                print("Reached ", int(widening / 0.01), " iterations of widening")
            new_cell = continuous_cell_selector(position, move)

    else:
        new_cell = [-1, -1]
        while not cell_is_reachable(position, new_cell, current_map):
            move = RANDOM_STATE.uniform(-1, 1, size=2)
            new_cell = continuous_cell_selector(position, move)
    return move


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


"""
def adapt_policy_nrpa(
    best_trajectory, best_course_of_actions, policy, score_change, learning_rate, sampling_method, relevance_radius_list
):
    if len(policy) > 0:
        cumulative_change = 0
        for position, previous_angle in policy.items():
            coefficients, values = list(), list()
            radius_index = 0
            while len(coefficients) == 0 and radius_index < len(relevance_radius_list):
                gaussian_filter = GaussianKernel(position, sigma=relevance_radius_list[radius_index])
                for point_index, point in enumerate(best_trajectory[:-1]):
                    if np.linalg.norm(np.array(list(position)) - np.array(list(point))) <= relevance_radius_list[radius_index]:
                        coefficients.append(gaussian_filter.pdf(point))
                        values.append(np.exp(complex(0, best_course_of_actions[point_index] * TWO_PI)))
                if coefficients:
                    coefficients = np.array(coefficients)
                    coefficients /= np.sum(coefficients)
                    assert not np.isnan(coefficients).any(), "NaN value found in coefficients computing"
                    new_angle = coefficients @ np.array(values).T
                radius_index += 1

            if len(coefficients) == 0 and radius_index >= len(relevance_radius_list):
                new_angle = np.exp(complex(0, TWO_PI * previous_angle))
            value_change = learning_rate * np.angle(new_angle - complex(0, TWO_PI *previous_angle)) / TWO_PI
            if np.isnan(value_change):
                raise ValueError("Value change at position " + str(position) + ": " + str(value_change))

            cumulative_change += value_change
            policy[position] = code_action(previous_angle + value_change)

        #print("Cumulative change: ", round(cumulative_change, 4))

        for point_index, point in enumerate(best_trajectory[:-1]):
            if code(point) not in policy.keys():
                policy[code(point)] = best_course_of_actions[point_index]
                cumulative_change = 100

    # If policy is empty (1st run)
    else:
        cumulative_change = 100
        for point_index, point in enumerate(best_trajectory[:-1]):
            policy[code(point)] = best_course_of_actions[point_index]

    # Reintegrating values in [0, 1] interval
    #for position, angle in policy.items():
    #    policy[position] = code_action(policy[position] % 1)

    # Checking range of normalized angles in the policy
    #for value in policy.values():
    #    if value > 1 or value < 0:
    #        raise ValueError("Angle outisde of [0, 1]: " + str(value))

    return policy
    """
