import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from utils.basic_functions import code_action
from utils.sampling_utils import *
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_SEED,
    N_GMM_COMPONENTS,
    RANDOM_STATE,
    TAU,
    N_SAMPLES_TO_CHOOSE_FROM,
    GAMMA,
)


def abgnrpa_step(position, goal, policy, heuristic_values, score) -> float:
    mixture_data = [
        RANDOM_STATE.multivariate_normal(
            np.array(list(key)),
            value,
            size=10,
        )
        for (key, value) in policy.items()
        if np.linalg.norm(np.array(key[:2]) - np.array(position)) <= RELEVANCE_RADIUS
    ]
    if mixture_data:

        model = GaussianMixture(N_GMM_COMPONENTS, random_state=RANDOM_SEED).fit(
            np.concatenate(mixture_data).reshape(-1, 3)
        )
        normalized_angles = sample_conditional_gmm_sklearn(
            model, list(position), n_samples=N_SAMPLES_TO_CHOOSE_FROM
        )
        probabilities = np.array(
            [
                np.exp(
                    model.predict(
                        np.array([position[0], position[1], angle]).reshape((1, -1))
                    )[0]
                    / TAU
                    + heuristic_values.get(position, goal, 2 * np.pi * angle)
                )
                for angle in normalized_angles
            ]
        )
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
    else:
        normalized_angle = RANDOM_STATE.uniform(0, 1)
    return normalized_angle
