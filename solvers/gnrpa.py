import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code_action
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    N_GMM_COMPONENTS,
    RANDOM_STATE,
    TAU,
    N_SAMPLES_TO_CHOOSE_FROM,
)


def compute_heuristic_value(position, goal, angle) -> float:
    vector = np.array(list(goal)) - np.array(list(position))
    return (vector @ np.array([np.cos(angle), np.sin(angle)]).T) / np.linalg.norm(
        vector
    )


def gnrpa_step(
    position, goal, current_map, policy, sampling_method: str = "GaussianMixture", sampling_radius=RELEVANCE_RADIUS / 4
) -> float:
    relevant_points = {
        key: value
        for (key, value) in policy.items()
        if np.linalg.norm(np.array(key[:2]) - np.array(position)) <= RELEVANCE_RADIUS
    }
    #print(policy)
    if len(relevant_points) > 0:
        if sampling_method == "GaussianMixture":
            mixture_data = [
                RANDOM_STATE.multivariate_normal(
                    np.array(list(key)),
                    value,
                    size=10,
                )
                for (key, value) in relevant_points.items()
            ]

            model = GaussianMixture(N_GMM_COMPONENTS, random_state=RANDOM_SEED).fit(
                np.concatenate(mixture_data).reshape(-1, 3)
            )
            normalized_angles = sample_conditional_gmm_sklearn(
                model, list(position), n_samples=N_SAMPLES_TO_CHOOSE_FROM
            )
            weights = [
                model.predict(
                    np.array([position[0], position[1], angle]).reshape((1, -1))
                )[0]
                for angle in normalized_angles
            ]

        else:
            x_data = np.zeros((len(relevant_points), 2))
            y_data = np.zeros((len(relevant_points)))
            for counter, (key, value) in enumerate(relevant_points.items()):
                x_data[counter, :] = np.array(list(key)).reshape((1, 2))
                y_data[counter] = value

            scaler = StandardScaler()
            x_data = scaler.fit_transform(x_data)
            model = get_model(sampling_method)
            model.fit(x_data, y_data)
            y_prediction = model.predict(x_data)
            residuals = y_data - y_prediction

            # Unbiased estimator
            sigma = max(np.sqrt(np.sum(residuals**2) / (len(y_data))), sampling_radius)
            #print(sigma)
            mean = model.predict(np.array(list(position)).reshape(1, -1))[0]
            normalized_angles = list()
            while len(normalized_angles) == 0:
                normalized_angles = RANDOM_STATE.normal(
                    mean, sigma, size=N_SAMPLES_TO_CHOOSE_FROM
                )
                normalized_angles = [
                    angle
                    for angle in normalized_angles
                    if cell_is_reachable(cell_selector(position, angle), current_map)
                ]
            weights = np.array([
                (
                    -0.5 * np.log(2 * np.pi)
                    - np.log(sigma)
                    - 0.5 * ((angle - mean) / sigma) ** 2
                )
                for angle in normalized_angles
            ])
        

    else:
        normalized_angles = list()
        while len(normalized_angles) == 0:
            normalized_angles = RANDOM_STATE.uniform(0, 1, size=N_SAMPLES_TO_CHOOSE_FROM)
            candidates = [
                cell_selector(position, angle)
                for angle in normalized_angles
            ]
            normalized_angles = [
                angle
                for angle in normalized_angles
                if cell_is_reachable(cell_selector(position, angle), current_map)
            ]
        weights = np.ones(len(normalized_angles)) / len(normalized_angles)
    heuristic_values = np.array([
        compute_heuristic_value(position, goal, 2 * np.pi * angle)
        for angle in normalized_angles
    ])
    heuristic_values = heuristic_values / np.sum(np.absolute(heuristic_values))
    probabilities = np.exp(
        weights / TAU + heuristic_values * 10
    )
    probabilities /= np.sum(probabilities)
    # Printing for debug
    for angle_number, angle in enumerate(normalized_angles):
        if np.isnan(weights[angle_number]) or np.isnan(heuristic_values[angle_number]) or np.isnan(probabilities[angle_number]):
            print("Angle ", angle, " has weight ", weights[angle_number], ", bias ", heuristic_values[angle_number], " and thus probability ", probabilities[angle_number])
            print(policy)
    for angle in normalized_angles:
        new_cell = cell_selector(position, angle)
        if not cell_is_reachable(new_cell, current_map):
            print("Error found here: ")
            print("Position: ", position)
            print("Angle: ", angle)
    return np.random.choice(normalized_angles, size=1, p=probabilities, replace=False)[
        0
    ]


def adapt_policy_gnrpa(
    best_trajectory, best_course_of_actions, policy, score_change, sampling_method
):
    for point_index, point in enumerate(best_trajectory[:-1]):
        i, j = point[0], point[1]
        if policy:
            restricted_policy = {
                key: value
                for key, value in policy.items()
                if np.linalg.norm(np.array(key[:2]) - np.array(point))
                <= RELEVANCE_RADIUS
            }
            if sampling_method == "GaussianMixture":
                average_covariance = np.mean(
                    np.array(list(restricted_policy.values())), axis=0
                )
                for key, value in restricted_policy.items():
                    policy[key] = update_covariance(
                        value,
                        np.array(key),
                        np.array(
                            [
                                i,
                                j,
                                code_action(best_course_of_actions[point_index]),
                            ]
                        ),
                        score_change,
                    )
                    restricted_policy[key] = policy[key]
                if (
                    policy.get(
                        (i, j, code_action(best_course_of_actions[point_index])),
                        None,
                    )
                    is None
                ):
                    policy[(i, j, code_action(best_course_of_actions[point_index]))] = (
                        average_covariance * 9 / 10
                    )

                normalizing_factor = np.sum(np.array(list(policy.values())), axis=0)
                for key, value in policy.items():
                    policy[key] = value / normalizing_factor
            else:
                for position, previous_angle in restricted_policy.items():
                    variation = (
                        LEARNING_RATE
                        * min((1 / np.linalg.norm(np.array(position) - np.array(point))), 1)
                        * (best_course_of_actions[point_index] - previous_angle)
                    )
                    #print(variation)
                    policy[position] += variation
                if (i, j) not in policy.keys():
                    policy[(i, j)] = best_course_of_actions[point_index]

        else:
            if sampling_method == "GaussianMixture":
                random_matrix = np.array(
                    [
                        [RANDOM_STATE.uniform(0, 1, size=(1, 1)) for __ in range(3)]
                        for _ in range(3)
                    ]
                ).reshape((3, 3))
                policy[(i, j, code_action(best_course_of_actions[point_index]))] = (
                    random_matrix @ random_matrix.T
                )
            else:
                policy[(i, j)] = best_course_of_actions[point_index]
    return policy
