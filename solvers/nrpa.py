import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from utils.basic_functions import code_action
from utils.sampling_utils import *
from utils.constants import RELEVANCE_RADIUS, RANDOM_SEED, N_GMM_COMPONENTS, RANDOM_STATE

def nrpa_step(position, policy):
    mixture_data = [
        RANDOM_STATE.multivariate_normal(
            np.array(list(key)),
            value,
            size=10,
        )
        for (key, value) in policy.items()
        if np.linalg.norm(np.array(key[:2]) - np.array(position))
        <= RELEVANCE_RADIUS
    ]
    if mixture_data:

        model = GaussianMixture(N_GMM_COMPONENTS, random_state=RANDOM_SEED).fit(
            np.concatenate(mixture_data).reshape(-1, 3)
        )
        normalized_angle = sample_conditional_gmm_sklearn(model, list(position))
    else:
        normalized_angle = RANDOM_STATE.uniform(0, 1)
    return normalized_angle

def adapt_policy(
    best_trajectory, best_course_of_actions, policy, best_score
):
    best_score = np.exp(-best_score)
    for point_index, point in enumerate(best_trajectory[:-1]):
        i, j = point[0], point[1]
        if policy:
            restricted_policy = {
                key: value
                for key, value in policy.items()
                if np.linalg.norm(np.array(key[:2]) - np.array(point))
                <= RELEVANCE_RADIUS
            }
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
                    best_score,
                )
                restricted_policy[key] = policy[key]
            if (
                policy.get(
                    (i, j, code_action(best_course_of_actions[point_index])),
                    None,
                )
                is None
            ):
                policy[
                    (i, j, code_action(best_course_of_actions[point_index]))
                ] = (average_covariance * 9 / 10)

            normalizing_factor = np.sum(
                np.array(list(policy.values())), axis=0
            )
            for key, value in policy.items():
                policy[key] = value / normalizing_factor

        else:
            random_matrix = np.array(
                [
                    [RANDOM_STATE.uniform(0, 1, size=(1, 1)) for __ in range(3)]
                    for _ in range(3)
                ]
            ).reshape((3, 3))
            policy[
                (i, j, code_action(best_course_of_actions[point_index]))
            ] = (random_matrix @ random_matrix.T)
    return policy
