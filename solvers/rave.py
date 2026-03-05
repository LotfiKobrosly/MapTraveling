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


def compute_continuous_amaf(normalized_angle, new_cell, states_values, kernel=None):
    """
    We compute the AMAF value using a gaussian convolution as mentioned in :
    Romain Michelucci, Denis Pallez, Tristan Cazenave, Jean-Paul Comet. Improving continuous Monte
    Carlo Tree Search for Identifying Parameters in Hybrid Gene Regulatory Networks. Parallel Problem
    Solving From Nature, Sep 2024, Hagenberg Castle, Austria. pp.319-334, ff10.1007/978-3-031-70085-
    9_20ff. ffhal-04557914f
    """
    amaf_value = 0
    amaf_signal = list()
    angle = 2 * np.pi * normalized_angle
    for position, values in states_values.items():
        state_distance = np.linalg.norm(np.array(position) - np.array(list(new_cell)))
        if state_distance < RELEVANCE_RADIUS:
            if state_distance == 0:
                amaf_signal.append(0)
            else:
                for action in states_values[position]["children"].keys():
                    action_angle = 2 * np.pi * action
                    amaf_signal.append(
                        np.log(
                            state_distance**2 / STATE_DISTANCE_PARAMETER
                            + (angle - action_angle) ** 2
                            / ACTION_DISTANCE_PARAMETER
                        )
                        * states_values[position]["mean_score"]
                    )
    if amaf_signal:
        if kernel is None:
            kernel = gaussian_kernel(RELEVANCE_RADIUS / 2)
        amaf_value = np.convolve(amaf_signal, kernel, mode="same").sum()
    return amaf_value


def compute_continuous_pamaf(
    normalized_angle, new_cell, states_values, kernel=None
) -> float:
    pamaf_value = 0
    pamaf_signal = list()
    angle = 2 * np.pi * normalized_angle
    for position, values in states_values.items():
        state_distance = np.linalg.norm(np.array(position) - np.array(list(new_cell)))
        if state_distance < RELEVANCE_RADIUS:
            if state_distance == 0:
                pamaf_signal.append(0)
            else:
                for action in states_values[position]["children"].keys():
                    action_angle = 2 * np.pi * action
                    pamaf_signal.append(
                        np.log(
                            state_distance**2 / STATE_DISTANCE_PARAMETER
                            + (angle - action_angle) ** 2
                            / ACTION_DISTANCE_PARAMETER
                        )
                    )
    if pamaf_signal:
        if kernel is None:
            kernel = gaussian_kernel(RELEVANCE_RADIUS / 2)
        pamaf_value = np.convolve(pamaf_signal, kernel, mode="same").sum()
    return pamaf_value


def rave_selection(
    position: tuple,
    candidate_actions_positions: dict,
    states_values: dict,
    actions_values: dict,
    continuous: bool = False,
    grave: bool = False,
    n_visits_reference: int = N_VISITS_REFERENCE,
    reference_position: tuple = None,
) -> tuple:
    chosen_angle = None
    best_blended_value = -np.inf
    for angle, state in candidate_actions_positions.items():
        if grave:
            if states_values[tuple(position)]["n_visits"] > n_visits_reference:
                reference_position = position
        else:
            reference_position = position
        uct_value = compute_uct(position, state, states_values)
        if continuous:
            amaf_value = compute_continuous_amaf(angle, state, states_values)
            pamaf_value = compute_continuous_pamaf(angle, state, states_values)
            reference_pamaf_value = compute_continuous_pamaf(
                angle, reference_position, states_values
            )
            #print("AMAF: ", amaf_value)
            #print("pAMAF: ", pamaf_value)
            #print("Reference pAMAF: ", reference_pamaf_value)
        else:
            amaf_value = get_discrete_amaf(position, angle, actions_values)
            pamaf_value = states_values[tuple(state)]["n_visits"]
            reference_pamaf_value = get_discrete_pamaf(
                reference_position, angle, actions_values
            )
        beta = compute_beta(
            reference_pamaf_value,
            pamaf_value,
        )
        blended_value = ((1 - beta) * uct_value + beta * amaf_value)
        #print(blended_value)
        if  blended_value > best_blended_value:
            chosen_angle = angle
            best_blended_value = blended_value

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
