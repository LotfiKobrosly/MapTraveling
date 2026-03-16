import numpy as np

from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code_action, code_position
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    N_GMM_COMPONENTS,
    RANDOM_STATE,
    LEARNING_RATE,
    EPSILON,
    N_EPOCHS
)


def nrpa_step(
    position,
    current_map,
    policy,
    sampling_radius=0.001,
    relevance_radius_list=[RELEVANCE_RADIUS],
):
    if len(policy) > 1:

        # Unbiased estimator
        #mean = model.predict(np.array(list(position)).reshape(1, -1))[0]
        if position in policy.keys():
            mean = policy[position]
        else:
            coefficients = np.zeros((len(policy)))
            angles = np.zeros((len(policy)))
            radius_index = 0
            while np.sum(coefficients) < 0.5 and radius_index < len(relevance_radius_list):
                gaussian_filter = GaussianKernel(position, sigma=relevance_radius_list[radius_index])
                for counter, (key, value) in enumerate(policy.items()):
                    coefficients[counter] = gaussian_filter.pdf(key)
                    angles[counter] = value
                radius_index += 1
            coefficients /= np.sum(coefficients)
            mean = angles @ coefficients.T
        new_cell = [-1, -1]
        widening = 0.01 # to prevent the search of being stuck
        while not cell_is_reachable(new_cell, current_map):
            normalized_angle = RANDOM_STATE.normal(mean, sampling_radius + widening, size=1)[0]
            widening += 0.01
            if int(widening / 0.01) % 100 == 0:
                print("Mean ", mean)
                print("Sigma: ", sampling_radius + widening_factor)
                print("Reached ", int(widening / 0.01), " iterations of widening")
            new_cell = continuous_cell_selector(position, normalized_angle)

    else:
        new_cell = [-1, -1]
        while not cell_is_reachable(new_cell, current_map):
            normalized_angle = RANDOM_STATE.uniform(0, 1, size=1)[0]
            new_cell = continuous_cell_selector(position, normalized_angle)
    return normalized_angle


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
                        values.append(best_course_of_actions[point_index])
                if coefficients:
                    coefficients = np.array(coefficients)
                    coefficients /= np.sum(coefficients)
                    assert not np.isnan(coefficients).any(), "NaN value found in coefficients computingq"
                    new_angle = coefficients @ np.array(values).T
                radius_index += 1

            else:
                new_angle = previous_angle
            value_change = learning_rate * ((new_angle - previous_angle) % 1)
            if np.isnan(value_change):
                raise ValueError("Value change at position " + str(position) + ": " + str(value_change))

            cumulative_change += value_change
            policy[position] += value_change # * score_change

        #print("Cumulative change: ", round(cumulative_change, 4))

        for point_index, point in enumerate(best_trajectory[:-1]):
            if code_position(point) not in policy.keys():
                policy[code_position(point)] = best_course_of_actions[point_index]
                cumulative_change = 100

    # If policy is empty (1st run)
    else:
        cumulative_change = 100
        for point_index, point in enumerate(best_trajectory[:-1]):
            policy[code_position(point)] = best_course_of_actions[point_index]

    # Reintegrating values in [0, 1] interval
    for position, angle in policy.items():
        policy[position] = code_action(policy[position] % 1)

    # Checking range of normalized angles in the policy
    for value in policy.values():
        if value > 1 or value < 0:
            raise ValueError("Angle outisde of [0, 1]: " + str(value))

    return policy
