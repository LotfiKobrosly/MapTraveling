# This file implements Rapid Action Value Estimation
import random
import numpy as np
from utils.basic_functions import gaussian_kernel, cosine_similarity
from utils.constants import (
    RELEVANCE_RADIUS,
    STATE_DISTANCE_PARAMETER,
    ACTION_DISTANCE_PARAMETER,
    BIAS_VALUE,
    DISCRETE_ACTIONS,
    N_VISITS_REFERENCE,
)
from solvers.mcts import discrete_possible_moves, compute_uct


def get_discrete_amaf(position: tuple, angle: float, actions_values: dict) -> float:
    if actions_values[tuple(position)][angle]["n_visits"] == 0:
        return 0
    else:
        return (
            actions_values[tuple(position)][angle]["cumulative_score"]
            / actions_values[tuple(position)][angle]["n_visits"]
        )


def get_discrete_pamaf(position: tuple, angle: float, actions_values: dict) -> int:
    return actions_values[tuple(position)][angle]["n_visits"]


def compute_beta(parent_pamaf_value: float, child_pamaf_value: float) -> float:
    return child_pamaf_value / (
        child_pamaf_value
        + parent_pamaf_value
        + BIAS_VALUE * child_pamaf_value * parent_pamaf_value
    )


def discrete_rave_selection(
    position: tuple,
    candidate_actions_positions: dict,
    states_values: dict,
    actions_values: dict,
    grave: bool = False,
    n_visits_reference: int = N_VISITS_REFERENCE,
    reference_position: tuple = None,
) -> tuple:
    chosen_angle = None
    best_blended_value = -np.inf
    for angle, state in candidate_actions_positions.items():
        uct_value = compute_uct(position, state, states_values)
        amaf_value = get_discrete_amaf(position, angle, actions_values)
        if grave:
            if states_values[tuple(position)]["n_visits"] > n_visits_reference:
                reference_position = position
        else:
            reference_position = position
        beta = compute_beta(
            get_discrete_pamaf(reference_position, angle, actions_values),
            states_values[tuple(state)]["n_visits"],
        )
        if ((1 - beta) * uct_value + beta * amaf_value) > best_blended_value:
            chosen_angle = angle

    return chosen_angle, candidate_actions_positions.get(chosen_angle, None)


def discrete_rave_expansion(position: tuple, states_values: dict):
    """
    Only callable when states_values[position]["unvisited_children"] is non-empty
    """
    angle = random.choice(
        list(states_values[tuple(position)]["unvisited_children"].keys()),
    )
    return angle, states_values[tuple(position)]["unvisited_children"][angle]


def discrete_rave_simulation(
    position: tuple, current_map: np.ndarray, actions_values: dict
):
    candidates = discrete_possible_moves(position, current_map)
    chosen_angle = np.random.choice(list(candidates.keys()), size=1)[0]
    return chosen_angle, candidates[chosen_angle]


def rave_backpropagation(actions_list: list, actions_values: dict, score: float):
    for position in actions_values.keys():
        for action in actions_values[position].keys():
            if action in actions_list:
                actions_values[position][action]["cumulative_score"] -= score


def compute_continuous_amaf(position, new_position, states_values, kernel=None):
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
                        np.array(key) - np.array(position),
                    )
                    ** 2
                    / ACTION_DISTANCE_PARAMETER
                )
                * states_values[key]["mean_score"]
            )
    if amaf_signal:
        if kernel is None:
            kernel = gaussian_kernel(RELEVANCE_RADIUS / 2)
        amaf_value = np.convolve(amaf_signal, kernel, mode="same").sum()
    return amaf_value


def compute_continuous_pamaf(
    position, new_position, states_values, kernel=None
) -> float:
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
                        np.array(key) - np.array(position),
                    )
                    ** 2
                    / ACTION_DISTANCE_PARAMETER
                )
            )
    if pamaf_signal:
        if kernel is None:
            kernel = gaussian_kernel(RELEVANCE_RADIUS / 2)
        pamaf_value = np.convolve(pamaf_signal, kernel, mode="same").sum()
    return pamaf_value


def compute_rave_uct(
    current_position: list,
    candidate_position: list,
    states_values: dict,
    is_continuous: bool = False,
) -> float:
    if is_continuous:
        amaf_value = compute_continuous_amaf(
            current_position, candidate_position, states_values
        )
        pamaf_value = compute_continuous_pamaf(
            current_position, candidate_position, states_values
        )

    else:
        amaf_value = compute_discrete_amaf(
            current_position, candidate_position, states_values
        )
        pamaf_value = compute_discrete_pamaf(
            current_position, candidate_position, states_values
        )
    beta = compute_beta(states_values[tuple(current_position)]["n_visits"], pamaf_value)
    return (1 - beta) * states_values[tuple(candidate_position)][
        "mean_score"
    ] + beta * amaf_value


def compute_grave_uct(
    current_position: list,
    candidate_position: list,
    reference_position: list,
    states_values: dict,
    is_continuous: bool = False,
) -> float:
    if is_continuous:
        amaf_value = compute_continuous_amaf(
            reference_position, candidate_position, states_values
        )
        pamaf_value = compute_continuous_pamaf(
            reference_position, candidate_position, states_values
        )

    else:
        amaf_value = compute_discrete_amaf(
            reference_position, candidate_position, states_values
        )
        pamaf_value = compute_discrete_pamaf(
            reference_position, candidate_position, states_values
        )
    beta = compute_beta(
        states_values[tuple(reference_position)]["n_visits"], pamaf_value
    )
    return (1 - beta) * states_values[tuple(candidate_position)][
        "mean_score"
    ] + beta * amaf_value
