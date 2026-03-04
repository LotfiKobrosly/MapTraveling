import random
import numpy as np

from utils.map_utils import cell_selector, get_intermediary_passage_points
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


def discrete_random_simulation(position: tuple, current_map: np.ndarray) -> tuple:
    return random.choice(list(discrete_possible_moves(position, current_map).values()))


def discrete_selection(
    position: tuple, candidate_positions: list, states_values: dict
) -> tuple:
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


def continuous_random_simulation(position: tuple, current_map: np.ndarray) -> tuple:
    height, width = current_map.shape
    next_cell_found = False
    while not next_cell_found:
        angle = RANDOM_STATE.uniform(0, 1)
        new_cell = cell_selector(position, angle * 2 * np.pi)
        if (
            (new_cell[0] >= 0)
            and (new_cell[0] < height)
            and (new_cell[1] >= 0)
            and (new_cell[1] < width)
            and (current_map[new_cell[0], new_cell[1]] != 1)
        ):
            next_cell_found = True
            intermediary_passage_points = get_intermediary_passage_points(
                position, new_cell, current_map
            )
            for point in intermediary_passage_points:
                if current_map[point[0], point[1]] == 1:
                    next_cell_found = False
                    break
            if not next_cell_found:
                continue
            break
    return angle, new_cell


def backpropagation(trajectory: list, states_values: dict, score: float):
    for position in trajectory:
        if not states_values.get(tuple(position), None) is None:
            states_values[tuple(position)]["cumulative_score"] -= score
            states_values[tuple(position)]["mean_score"] = (
                states_values[tuple(position)]["cumulative_score"]
                / states_values[tuple(position)]["n_visits"]
            )
