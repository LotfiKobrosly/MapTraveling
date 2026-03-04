import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.basic_functions import code_action, cosine_similarity
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, cell_selector
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    N_GMM_COMPONENTS,
    RANDOM_STATE,
    LEARNING_RATE,
    EPSILON,
)


def nrpa_step(
    position,
    current_map,
    policy,
    sampling_method: str = "GaussianMixture",
    sampling_radius=RELEVANCE_RADIUS / 4,
):
    relevant_points = {
        key: value
        for (key, value) in policy.items()
        if np.linalg.norm(np.array(key[:2]) - np.array(position)) <= RELEVANCE_RADIUS
    }
    if len(relevant_points) > 0:
        # print(relevant_points)
        if sampling_method == "GaussianMixture":
            print("GaussianMixture")
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
            normalized_angle = sample_conditional_gmm_sklearn(model, list(position))
        else:
            # print(sampling_method)
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
            mean = model.predict(np.array(list(position)).reshape(1, -1))[0]
            new_cell = [-1, -1]
            while not cell_is_reachable(new_cell, current_map):
                normalized_angle = RANDOM_STATE.normal(mean, sigma, size=1)
                new_cell = cell_selector(position, normalized_angle)

    else:
        new_cell = [-1, -1]
        while not cell_is_reachable(new_cell, current_map):
            normalized_angle = RANDOM_STATE.uniform(0, 1, size=1)
            new_cell = cell_selector(position, normalized_angle)
    return normalized_angle


def adapt_policy_nrpa(
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
                    policy[position] += (
                        LEARNING_RATE
                        * min(
                            (1 / np.linalg.norm(np.array(position) - np.array(point))),
                            1,
                        )
                        * (best_course_of_actions[point_index] - previous_angle)
                    )
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
