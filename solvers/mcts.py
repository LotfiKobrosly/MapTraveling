import random
import numpy as np

from utils.map_utils import (
    cell_selector,
    continuous_cell_selector,
    cell_is_reachable,
    get_intermediary_passage_points,
)
from utils.constants import (
    EXPLORATION_CONSTANT,
    RANDOM_STATE,
    STEP_SIZE,
    DISCRETE_ACTIONS,
)


def compute_uct(position: tuple, new_position: tuple, states_values: dict) -> float:
    return states_values[tuple(new_position)][
        "mean_score"
    ] + EXPLORATION_CONSTANT * np.sqrt(
        np.log(states_values[tuple(position)]["n_visits"])
        / states_values[tuple(new_position)]["n_visits"]
    )


def discrete_possible_moves(position: tuple, current_map: np.ndarray) -> dict:
    height, width = current_map.shape
    candidate_positions = [cell_selector(position, angle) for angle in DISCRETE_ACTIONS]
    actions_states_dict = {
        DISCRETE_ACTIONS[cell_id]: cell
        for cell_id, cell in enumerate(candidate_positions)
        if (cell[0] >= 0)
        and (cell[1] >= 0)
        and (cell[0] < height)
        and (cell[1] < width)
        and (current_map[*cell] != 1)
    }
    return actions_states_dict


def selection(position: tuple, candidate_positions: list, states_values: dict) -> tuple:
    uct_values = [
        compute_uct(position, child, states_values) for child in candidate_positions
    ]
    if uct_values:

        return candidate_positions[np.argmax(uct_values)]
    else:
        return None


def discrete_expansion(position: tuple, states_values: dict) -> tuple:
    """
    Only callable when states_values[position]["unvisited_children"] is non-empty
    """
    return random.choice(
        states_values[tuple(position)]["unvisited_children"],
    )


def discrete_random_simulation(position: tuple, current_map: np.ndarray) -> tuple:
    return random.choice(list(discrete_possible_moves(position, current_map).values()))


def continuous_random_simulation(position: tuple, current_map: np.ndarray) -> tuple:
    new_cell = (-1, -1)
    while not cell_is_reachable(new_cell, current_map):
        angle = RANDOM_STATE.uniform(0, 1)
        new_cell = continuous_cell_selector(position, angle)

    return angle, new_cell


def continuous_expansion(
    position: tuple, states_values: dict, current_map: np.ndarray
) -> tuple:
    angle, new_cell = continuous_random_simulation(position, current_map)
    while new_cell in states_values[tuple(position)]["children"]:
        angle, new_cell = continuous_random_simulation(position, current_map)

    return angle, new_cell


def backpropagation(trajectory: list, states_values: dict, score: float):
    for position in trajectory:
        if not states_values.get(tuple(position), None) is None:
            states_values[tuple(position)]["cumulative_score"] -= score
            states_values[tuple(position)]["mean_score"] = (
                states_values[tuple(position)]["cumulative_score"]
                / states_values[tuple(position)]["n_visits"]
            )
