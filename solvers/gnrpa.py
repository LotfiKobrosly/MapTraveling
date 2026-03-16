import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code_action, compute_heuristic_value, code_position
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
)


def gnrpa_step(
    position,
    goal,
    current_map,
    policy,
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
            widening = 0.01 # to prevent the search of being stuck
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
            compute_heuristic_value(position, goal, 2 * np.pi * angle)
            for angle in normalized_angles
        ]
    )
    biases_values = biases_values / np.sum(np.absolute(biases_values))
    probabilities = np.exp(weights / TAU + biases_values * 10)
    probabilities /= np.sum(probabilities)
    return np.random.choice(normalized_angles, size=1, p=probabilities, replace=False)[
        0
    ]


def adapt_policy_gnrpa(
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

    """
    # Refitting the model
    x_data = np.zeros((len(policy), 2))
    y_data = np.zeros((len(policy)))
    for counter, (key, value) in enumerate(policy.items()):
        x_data[counter, :] = np.array(list(key)).reshape((1, 2))
        y_data[counter] = value

    #scaler = StandardScaler()
    #x_data = scaler.fit_transform(x_data)
    #for _ in range(n_epochs):
    #model.fit(x_data, y_data)
    """
    return policy
