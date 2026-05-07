from copy import deepcopy
import random
import numpy as np

from classes.path_generator import PathGenerator
from solvers.mcts import continuous_random_simulation
from utils.basic_functions import code
from utils.map_utils import cell_is_reachable, continuous_cell_selector


def cnmcts(path_generator: PathGenerator, level: int = 1, bandwidth: int = 50):
    if level == 0:
        while not path_generator.is_finished():
            if (
                code(path_generator.current_position)
                in path_generator.states_actions.keys()
            ):
                move = random.choice(
                    path_generator.states_actions[code(path_generator.current_position)]
                )
                new_cell = continuous_cell_selector(
                    path_generator.current_position, move
                )
            else:
                move, new_cell = continuous_random_simulation(
                    path_generator.current_position, path_generator.current_map
                )
            path_generator.update(move, new_cell)
        return (
            path_generator.trajectory,
            path_generator.actions,
            path_generator.get_score(),
        )

    else:
        while not path_generator.is_finished():
            # scores_list = list()
            if (
                not code(path_generator.current_position)
                in path_generator.states_actions.keys()
            ):
                path_generator.states_actions[code(path_generator.current_position)] = (
                    list()
                )
                for _ in range(bandwidth):
                    new_cell = [-1, -1]
                    # visited_state = False
                    while not cell_is_reachable(
                        path_generator.current_position,
                        new_cell,
                        path_generator.current_map,
                    ):
                        move, new_cell = continuous_random_simulation(
                            path_generator.current_position, path_generator.current_map
                        )
                    path_generator.states_actions[
                        code(path_generator.current_position)
                    ].append(move)
            moves_list = path_generator.states_actions[
                code(path_generator.current_position)
            ]
            for move_index, move in enumerate(moves_list):
                move_scores_list = list()
                path = deepcopy(path_generator)
                path.update(move, continuous_cell_selector(path.current_position, move))
                trajectory, actions_list, score = cnmcts(path, level - 1, bandwidth)
                if score < path_generator.best_score:
                    path_generator.best_trajectory = trajectory[:]
                    path_generator.best_course_of_actions = actions_list[:]
                    path_generator.best_score = score
            # TODO: investigate the reason behind the need for the next if clause
            # It is not supposed to be needed
            if path_generator.current_steps >= len(
                path_generator.best_course_of_actions
            ):
                path_generator.actions = path_generator.best_course_of_actions[:][:-1]
                path_generator.trajectory = path_generator.best_trajectory[:][:-1]
                path_generator.current_steps = (
                    len(path_generator.best_course_of_actions) - 1
                )
            path_generator.update(
                path_generator.best_course_of_actions[path_generator.current_steps],
                path_generator.best_trajectory[path_generator.current_steps + 1],
            )

    return (
        path_generator.best_trajectory,
        path_generator.best_course_of_actions,
        path_generator.best_score,
    )


def crbnmcts(
    path_generator: PathGenerator,
    level: int = 1,
    bandwidth: int = 50,
    move: np.ndarray = None,
    new_position: tuple = None,
):
    if level == 0:
        assert not (move is None), "'move' must be specified for cRbNMCTS level 0"
        path_generator.update(move, new_position)
        return (
            path_generator.trajectory,
            path_generator.actions,
            np.linalg.norm(np.array(path_generator.goal) - np.array(new_position)),
        )

    else:
        if not (move is None):
            path_generator.update(move, new_position)
        while not path_generator.is_finished():
            # scores_list = list()
            if (
                not (
                    code(path_generator.current_position) in path_generator.states_actions.keys()
                )
            ):
                path_generator.states_actions[code(path_generator.current_position)] = (
                    list()
                )
                for _ in range(bandwidth):
                    new_cell = [-1, -1]
                    # visited_state = False
                    while not cell_is_reachable(
                        path_generator.current_position,
                        new_cell,
                        path_generator.current_map,
                    ):
                        move, new_cell = continuous_random_simulation(
                            path_generator.current_position, path_generator.current_map
                        )
                    path_generator.states_actions[
                        code(path_generator.current_position)
                    ].append(move)
            moves_list = path_generator.states_actions[
                code(path_generator.current_position)
            ]
            if level == 1:
                best_score = np.inf
            else:
                best_score = path_generator.best_score
            best_move = None
            best_new_cell = None
            for move_index, move in enumerate(moves_list):
                new_cell = continuous_cell_selector(
                    path_generator.current_position, move
                )
                path = deepcopy(path_generator)
                trajectory, actions_list, score = crbnmcts(
                    path,
                    level - 1,
                    bandwidth,
                    move,
                    new_cell,
                )
                if score < best_score:
                    best_score = score
                    best_move = deepcopy(move)
                    best_new_cell = deepcopy(new_cell)

                if path.is_finished() and (
                    path.get_score() < path_generator.best_score
                ):
                    path_generator.best_trajectory = trajectory[:]
                    path_generator.best_course_of_actions = actions_list[:]
                    path_generator.best_score = path.get_score()
            if best_move is None:
                best_move = path_generator.best_course_of_actions[
                    path_generator.current_steps
                ]
                best_new_cell = path_generator.best_trajectory[
                    path_generator.current_steps + 1
                ]
            path_generator.update(best_move, best_new_cell)

    return (
        path_generator.best_trajectory,
        path_generator.best_course_of_actions,
        path_generator.best_score,
    )
