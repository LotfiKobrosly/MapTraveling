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
    model,
    sampling_radius=RELEVANCE_RADIUS / 4,
):
    if len(policy) > 0:

        # Unbiased estimator
        #mean = model.predict(np.array(list(position)).reshape(1, -1))[0]
        if position in policy.keys():
            mean = policy[position]
        else:
            gaussian_filter = GaussianKernel2D(position, sigma=RELEVANCE_RADIUS)
            coefficients = np.zeros((len(policy)))
            angles = np.zeros((len(policy)))
            for counter, (key, value) in enumerate(policy.items()):
                coefficients[counter] = gaussian_filter.pdf(key)
                angles[counter] = value
            coefficients /= np.sum(coefficients)
            mean = angles @ coefficients.T
        new_cell = [-1, -1]
        widening = 1 # to prevent the search of being stuck
        while not cell_is_reachable(new_cell, current_map):
            normalized_angle = RANDOM_STATE.normal(mean, sampling_radius + widening, size=1)[0]
            widening += 0.01
            if int(widening / 0.01) % 100 == 0:
                print("Mean ", mean)
                print("Sigma: ", sampling_radius * widening_factor)
                print("Reached ", int(widening / 0.01), " iterations of widening")
            new_cell = continuous_cell_selector(position, normalized_angle)

    else:
        new_cell = [-1, -1]
        while not cell_is_reachable(new_cell, current_map):
            normalized_angle = RANDOM_STATE.uniform(0, 1, size=1)[0]
            new_cell = continuous_cell_selector(position, normalized_angle)
    return normalized_angle

#def fit_nrpa_model():


def adapt_policy_nrpa(
    best_trajectory, best_course_of_actions, policy, model, score_change, learning_rate, sampling_method, n_epochs=1
):
    if len(policy) > 0:
        cumulative_change = 0
        for position, previous_angle in policy.items():
            gaussian_filter = GaussianKernel2D(position, sigma=3)
            coefficients, values = list(), list()
            for point_index, point in enumerate(best_trajectory[:-1]):
                if np.linalg.norm(np.array(list(position)) - np.array(list(point))) <= RELEVANCE_RADIUS:
                    coefficients.append(gaussian_filter.pdf(point))
                    values.append(best_course_of_actions[point_index])
            if coefficients:
                coefficients = np.array(coefficients)
                new_angle = (coefficients / np.sum(coefficients)) @ np.array(values).T

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

    return policy, model, cumulative_change
