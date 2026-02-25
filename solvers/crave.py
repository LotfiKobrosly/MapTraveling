# This file implements Rapid Action Value Estimation

import numpy as np
from utils.basic_functions import gaussian_kernel, cosine_similarity
from utils.constants import (
    RELEVANCE_RADIUS,
    STATE_DISTANCE_PARAMETER,
    ACTION_DISTANCE_PARAMETER,
    BIAS_VALUE,
)


def compute_amaf(position, new_position, states_values, kernel=None):
    """
    We compute the AMAF value using a gaussian convolution as mentioned in :
    Romain Michelucci, Denis Pallez, Tristan Cazenave, Jean-Paul Comet. Improving continuous Monte
    Carlo Tree Search for Identifying Parameters in Hybrid Gene Regulatory Networks. Parallel Problem
    Solving From Nature, Sep 2024, Hagenberg Castle, Austria. pp.319-334, ff10.1007/978-3-031-70085-
    9_20ff. ffhal-04557914f
    """
    amaf_value = 0
    amaf_signal = list()
    for key, value in states_values.items():
        state_distance = np.linalg.norm(np.array(position) - np.array(list(key)))
        if state_distance < RELEVANCE_RADIUS:
            amaf_signal.append(
                np.log(
                    state_distance**2 / STATE_DISTANCE_PARAMETER
                    + cosine_similarity(
                        np.array(new_position) - np.array(position),
                        np.array(key) - np.array(position)
                    ) ** 2 / ACTION_DISTANCE_PARAMETER
                )
                * states_values[key]["mean_score"]
            )
    if amaf_signal:
        if kernel is None:
            kernel = gaussian_kernel(1)
        amaf_value = np.convolve(amaf_signal, kernel, mode="same").sum()
    return amaf_value

def compute_pamaf(position, new_position, states_values, kernel=None) -> float:
    pamaf_value = 0
    pamaf_signal = list()
    for key, value in states_values.items():
        state_distance = np.linalg.norm(np.array(position) - np.array(list(key)))
        if state_distance < RELEVANCE_RADIUS:
            pamaf_signal.append(
                np.log(
                    state_distance**2 / STATE_DISTANCE_PARAMETER
                    + cosine_similarity(
                        np.array(new_position) - np.array(position),
                        np.array(key) - np.array(position)
                    ) ** 2 / ACTION_DISTANCE_PARAMETER
                )
            )
    if pamaf_signal:
        if kernel is None:
            kernel = gaussian_kernel(1)
        pamaf_value = np.convolve(pamaf_signal, kernel, mode="same").sum()
    return pamaf_value

def compute_beta(parent_pamaf_value: float, child_pamaf_value: float) -> float:
    return child_pamaf_value / (child_pamaf_value + parent_pamaf_value + BIAS_VALUE * child_pamaf_value * parent_pamaf_value)

def compute_rave_uct(current_position: list, candidate_position: list, states_values: dict) -> float:
    amaf_value = compute_amaf(current_position, candidate_position, states_values)
    pamaf_value = compute_pamaf(current_position, candidate_position, states_values)
    beta = compute_beta(states_values[tuple(current_position)]["n_visits"], pamaf_value)
    return ((1 - beta) * states_values[tuple(candidate_position)]["mean_score"] + beta * amaf_value)

def compute_grave_uct(current_position: list, candidate_position: list, reference_position: list, states_values: dict) -> float:   
    amaf_value = compute_amaf(current_position, candidate_position, states_values)
    pamaf_value = compute_pamaf(current_position, candidate_position, states_values)
    beta = compute_beta(states_values[tuple(reference_position)]["n_visits"], pamaf_value)
    return (1 - beta) * states_values[tuple(candidate_position)]["mean_score"] + beta * amaf_value
