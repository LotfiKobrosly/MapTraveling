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
    TWO_PI,
)
from solvers.mcts import discrete_possible_moves, compute_uct


def get_discrete_amaf(position: tuple, move: tuple, actions_values: dict) -> float:
    if actions_values[tuple(position)][move]["n_visits"] == 0:
        return 0
    else:
        return (
            actions_values[tuple(position)][move]["cumulative_score"]
            / actions_values[tuple(position)][move]["n_visits"]
        )


def get_discrete_pamaf(position: tuple, move: tuple, actions_values: dict) -> int:
    return actions_values[tuple(position)][move]["n_visits"]


def compute_beta(parent_pamaf_value: float, child_pamaf_value: float) -> float:
    return child_pamaf_value / (
        child_pamaf_value
        + parent_pamaf_value
        + BIAS_VALUE * child_pamaf_value * parent_pamaf_value
    )


def compute_continuous_amaf(move, new_cell, states_values, kernel=None):
    """
    We compute the AMAF value using a modified version of a gaussian convolution as mentioned in:
    Romain Michelucci, Denis Pallez, Tristan Cazenave, Jean-Paul Comet. Improving continuous Monte
    Carlo Tree Search for Identifying Parameters in Hybrid Gene Regulatory Networks. Parallel Problem
    Solving From Nature, Sep 2024, Hagenberg Castle, Austria. pp.319-334, ff10.1007/978-3-031-70085-
    9_20ff. ffhal-04557914f
    """
    amaf_value = 0
    amaf_components = list()
    for position, values in states_values.items():
        state_distance = np.linalg.norm(np.array(position) - np.array(list(new_cell)))
        if state_distance < RELEVANCE_RADIUS:
            if state_distance == 0:
                amaf_components.append(0)
            else:
                for action in states_values[position]["children"].keys():
                    amaf_components.append(
                        np.log(
                            state_distance**2 / STATE_DISTANCE_PARAMETER
                            + np.linalg.norm(
                                np.array(list(action)) - np.array(list(move))
                            )
                            ** 2
                            / ACTION_DISTANCE_PARAMETER
                        )
                        * states_values[position]["mean_score"]
                    )
    if amaf_components:
        if kernel is None:
            kernel = gaussian_kernel(RELEVANCE_RADIUS / 2)
        amaf_value = np.convolve(amaf_components, kernel, mode="same").sum()
    return amaf_value


def compute_continuous_pamaf(move, new_cell, states_values, kernel=None) -> float:
    pamaf_value = 0
    pamaf_components = list()
    for position, values in states_values.items():
        state_distance = np.linalg.norm(np.array(position) - np.array(list(new_cell)))
        if state_distance < RELEVANCE_RADIUS:
            if state_distance == 0:
                pamaf_components.append(0)
            else:
                for action in states_values[position]["children"].keys():
                    pamaf_components.append(
                        np.log(
                            state_distance**2 / STATE_DISTANCE_PARAMETER
                            + np.linalg.norm(
                                np.array(list(action)) - np.array(list(move))
                            )
                            ** 2
                            / ACTION_DISTANCE_PARAMETER
                        )
                    )
    if pamaf_components:
        if kernel is None:
            kernel = gaussian_kernel(RELEVANCE_RADIUS / 2)
        pamaf_value = np.convolve(pamaf_components, kernel, mode="same").sum()
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
    chosen_move = None
    best_blended_value = -np.inf
    for move, state in candidate_actions_positions.items():
        if grave:
            if states_values[tuple(position)]["n_visits"] > n_visits_reference:
                reference_position = position
        else:
            reference_position = position
        uct_value = compute_uct(position, state, states_values)
        if continuous:
            amaf_value = compute_continuous_amaf(move, state, states_values)
            pamaf_value = compute_continuous_pamaf(move, state, states_values)
            reference_pamaf_value = compute_continuous_pamaf(
                move, reference_position, states_values
            )
            # print("AMAF: ", amaf_value)
            # print("pAMAF: ", pamaf_value)
            # print("Reference pAMAF: ", reference_pamaf_value)
        else:
            amaf_value = get_discrete_amaf(position, move, actions_values)
            pamaf_value = states_values[tuple(state)]["n_visits"]
            reference_pamaf_value = get_discrete_pamaf(
                reference_position, move, actions_values
            )
        beta = compute_beta(
            reference_pamaf_value,
            pamaf_value,
        )
        blended_value = (1 - beta) * uct_value + beta * amaf_value
        # print(blended_value)
        if blended_value > best_blended_value:
            chosen_move = move
            best_blended_value = blended_value

    return chosen_move, candidate_actions_positions.get(chosen_move, None)


def discrete_rave_expansion(position: tuple, states_values: dict):
    """
    Only callable when states_values[position]["unvisited_children"] is non-empty
    """
    move = random.choice(
        list(states_values[tuple(position)]["unvisited_children"].keys()),
    )
    return move, states_values[tuple(position)]["unvisited_children"][move]


def discrete_rave_simulation(
    position: tuple, current_map: np.ndarray, actions_values: dict
):
    candidates = discrete_possible_moves(position, current_map)
    possible_indices = list(range(len(candidates)))
    chosen_move = list(candidates.keys())[np.random.choice(possible_indices, size=1)[0]]
    return chosen_move, candidates[chosen_move]


def rave_backpropagation(actions_list: list, actions_values: dict, score: float):
    for position in actions_values.keys():
        for action in actions_values[position].keys():
            if action in actions_list:
                actions_values[position][action]["cumulative_score"] -= score
