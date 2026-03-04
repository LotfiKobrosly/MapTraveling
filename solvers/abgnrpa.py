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
    GAMMA,
)
from classes.heuristic_values import HeuristicValues


def abgnrpa_step(
    position: tuple,
    goal: tuple,
    current_map: np.ndarray,
    policy: dict,
    heuristic_values: HeuristicValues,
    score: float,
    sampling_method: str = "GaussianMixture",
    sampling_radius=RELEVANCE_RADIUS / 4,
) -> float:
    relevant_points = {
        key: value
        for (key, value) in policy.items()
        if np.linalg.norm(np.array(key[:2]) - np.array(position)) <= RELEVANCE_RADIUS
    }
    # print(policy)
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
            # print(sigma)
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
            weights = np.array(
                [
                    (
                        -0.5 * np.log(2 * np.pi)
                        - np.log(sigma)
                        - 0.5 * ((angle - mean) / sigma) ** 2
                    )
                    for angle in normalized_angles
                ]
            )

    else:
        normalized_angles = list()
        while len(normalized_angles) == 0:
            normalized_angles = RANDOM_STATE.uniform(
                0, 1, size=N_SAMPLES_TO_CHOOSE_FROM
            )
            candidates = [cell_selector(position, angle) for angle in normalized_angles]
            normalized_angles = [
                angle
                for angle in normalized_angles
                if cell_is_reachable(cell_selector(position, angle), current_map)
            ]
        weights = np.ones(len(normalized_angles)) / len(normalized_angles)
    biases = np.array(
        [heuristic_values.get(position, goal, angle) for angle in normalized_angles]
    )
    biases = biases / np.sum(np.absolute(biases))
    probabilities = np.exp(weights / TAU + biases * 10)
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
