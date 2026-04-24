"""
Agglomerates defined functions and runs solvers
"""

from copy import deepcopy
import numpy as np

from classes import *
from solvers.abgnrpa import *
from solvers.cnmcts import *
from solvers.gnrpa import *
from solvers.mcts import *
from solvers.nrpa import *
from solvers.pbrnrpa import *
from solvers.rave import *
from utils.basic_functions import code
from utils.map_utils import cell_is_reachable, continuous_cell_selector
from utils.constants import *


def step(path_generator: PathGenerator, policy: dict = None, relevance_radius_list: list = None, heuristic_values: HeuristicValues = None):
    path_generator.current_position = code(path_generator.current_position)
    if path_generator.strategy in ["nrpa", "gnrpa", "abgnrpa", "pbrnrpa"]:
        sampling_radius = np.exp(
            -path_generator.nrpa_iterations / (path_generator.n_policies * HALF_LIFE_DIVIDER)
        )
    if path_generator.strategy == "random_walk":
        return continuous_random_simulation(path_generator.current_position, path_generator.current_map)

    elif path_generator.strategy == "nrpa":
        assert not (policy is None), "For " + path_generator.strategy + ", 'policy' must be defined"
        assert not (relevance_radius_list is None), "For " + path_generator.strategy + ", 'relevance_radius_list' must be defined"
        move, new_cell = nrpa_step(
            path_generator.current_position,
            path_generator.current_map,
            policy,
            sampling_radius=sampling_radius,
            relevance_radius_list=relevance_radius_list,
        )

    elif path_generator.strategy == "gnrpa":
        assert not (policy is None), "For " + path_generator.strategy + ", 'policy' must be defined"
        assert not (relevance_radius_list is None), "For " + path_generator.strategy + ", 'relevance_radius_list' must be defined"
        move, new_cell = gnrpa_step(
            path_generator.current_position,
            path_generator.goal,
            path_generator.current_map,
            policy,
            sampling_radius=sampling_radius,
            relevance_radius_list=relevance_radius_list,
        )

    elif path_generator.strategy == "pbrnrpa":
        assert not (policy is None), "For " + path_generator.strategy + ", 'policy' must be defined"
        move, new_cell = pbrnrpa_step(
            path_generator.current_position,
            path_generator.current_map,
            policy,
            sampling_radius=sampling_radius,
        )

    elif path_generator.strategy == "abgnrpa":
        assert not (policy is None), "For " + path_generator.strategy + ", 'policy' must be defined"
        assert not (relevance_radius_list is None), "For " + path_generator.strategy + ", 'relevance_radius_list' must be defined"
        assert not (heuristic_values is None), "For ABGNRPA 'heuristic_values' must be defined"
        move, new_cell = abgnrpa_step(
            path_generator.current_position,
            path_generator.goal,
            path_generator.current_map,
            policy,
            heuristic_values,
            1 - path_generator.get_score() / path_generator.score_normalizer,
            sampling_radius=sampling_radius,
            relevance_radius_list=relevance_radius_list,
        )

    else:
        raise (ValueError("No valid strategy defined"))

    return code(move), new_cell

def generate_path(path_generator: PathGenerator, policy: dict = None, relevance_radius_list: list = None, heuristic_values: HeuristicValues = None):

    path_generator.trajectory = [path_generator.start_point]
    while not path_generator.is_finished():
        move, new_cell = step(path_generator, policy, relevance_radius_list, heuristic_values)
        assert cell_is_reachable(
            path_generator.current_position, new_cell, path_generator.current_map
        ), ("Position " + str(new_cell) + " out of bounds OR inside obstacle")
        path_generator.update(move, new_cell)

def adapt_policy(
    strategy,
    best_trajectory,
    best_course_of_actions,
    policy,
    learning_rate,
):
    if strategy == "nrpa":
        return adapt_policy_nrpa(
            best_trajectory,
            policy,
            learning_rate,
        )
    elif strategy == "pbrnrpa":
        return adapt_pbrnrpa_policy(
            best_trajectory,
            best_course_of_actions,
            policy,
            learning_rate,
        )
    elif strategy in ["gnrpa", "abgnrpa"]:
        return adapt_policy_gnrpa(
            best_trajectory,
            policy,
            learning_rate,
        )
    else:
        raise ValueError("Wrong strategy for policy adaptation")


def nrpa(
    path_generator: PathGenerator,
    level: int = 1,
    n_policies: int = 100,
    policy: dict = dict(),
    relevance_radius_list: list = None,
    heuristic_values: HeuristicValues = None,
):
    path_generator.n_policies = n_policies
    if level == 0:
        path_generator.reinitialize()
        path_generator.nrpa_iterations += 1
        generate_path(path_generator, policy, relevance_radius_list, heuristic_values)
        score = path_generator.get_score()
        if score < path_generator.best_score:
            path_generator.best_score = score
            path_generator.best_trajectory = path_generator.trajectory[:]
            path_generator.best_course_of_actions = path_generator.actions[:]

    else:
        iteration_number = path_generator.nrpa_iterations
        best_score = path_generator.best_score
        last_best_score = best_score
        best_trajectory = deepcopy(path_generator.trajectory)
        best_course_of_actions = deepcopy(path_generator.actions)
        new_policy = deepcopy(policy)
        learning_rate = np.sqrt(1 / path_generator.trajectory_size)
        path_generator.cumulative_change = 100
        for iteration_number in range(n_policies):
            nrpa(path_generator, level - 1, n_policies, policy, relevance_radius_list, heuristic_values)
            score = path_generator.get_score()
            if score < best_score:
                best_score, score = score, best_score
                best_trajectory = deepcopy(path_generator.trajectory)
                best_course_of_actions = deepcopy(path_generator.actions)
                print(
                    "Better score found at iteration ",
                    iteration_number + 1,
                    ": ",
                    int(best_score),
                )
            score = path_generator.get_score()
            if (iteration_number + 1) % 100 == 0:
                print(
                    "Iteration n° ",
                    iteration_number + 1,
                    ": best score: ",
                    best_score,
                )
            if path_generator.strategy == "pbrnrpa":
                existing_regions = list(new_policy.keys())
                for bounding_coordinates in existing_regions:
                    if (
                        new_policy[bounding_coordinates]["n_visits"]
                        > new_policy[bounding_coordinates]["threshold"]
                    ):
                        for new_region in subdivide_region(bounding_coordinates):
                            new_policy[new_region] = {
                                "move": new_policy[bounding_coordinates]["move"],
                                "n_visits": new_policy[bounding_coordinates][
                                    "n_visits"
                                ],
                                "threshold": int(
                                    path_generator.area / get_region_area(new_region)
                                ),
                            }
                        del new_policy[bounding_coordinates]
            new_policy = adapt_policy(
                path_generator.strategy,
                best_trajectory,
                best_course_of_actions,
                new_policy,
                learning_rate,
            )
        path_generator.nrpa_iterations = iteration_number + 1
        policy = adapt_policy(
            path_generator.strategy,
            best_trajectory,
            best_course_of_actions,
            policy=policy,
            learning_rate=learning_rate,
        )
        path_generator.trajectory = deepcopy(best_trajectory)
        path_generator.best_course_of_actions = deepcopy(best_course_of_actions)
        path_generator.best_score = best_score

def mcts(path_generator, n_iterations: int = 10000):
    """
    Discrete MCTS
    """
    best_trajectory = None
    best_score = path_generator.best_score
    if path_generator.strategy == "mcts":
        path_generator.states_values[tuple(path_generator.current_position)]["unvisited_children"] = (
            list(
                discrete_possible_moves(
                    path_generator.current_position, path_generator.current_map
                ).values()
            )
        )
    elif path_generator.strategy in ["rave", "grave"]:
        path_generator.states_values[tuple(path_generator.current_position)]["unvisited_children"] = (
            discrete_possible_moves(path_generator.current_position, path_generator.current_map)
        )
    for iteration_number in range(n_iterations):
        path_generator.reinitialize()
        visited_states = {tuple(path_generator.start_point)}
        reference_position = path_generator.start_point

        # Selection
        selection_length = 0
        no_cell_found = False
        while (
            len(
                path_generator.states_values[tuple(path_generator.current_position)][
                    "unvisited_children"
                ]
            )
            == 0
        ) and not path_generator.is_finished():
            path_generator.states_values[tuple(path_generator.current_position)]["n_visits"] += 1
            if path_generator.strategy == "mcts":
                new_cell = selection(
                    path_generator.current_position,
                    [
                        child
                        for child in path_generator.states_values[
                            tuple(path_generator.current_position)
                        ]["children"]
                        if child not in visited_states
                    ],
                    path_generator.states_values,
                )
                normalized_angle = None
            elif path_generator.strategy in ["rave", "grave"]:
                grave = False
                if path_generator.strategy == "grave":
                    grave = True
                normalized_angle, new_cell = rave_selection(
                    path_generator.current_position,
                    {
                        angle: child
                        for angle, child in path_generator.states_values[
                            tuple(path_generator.current_position)
                        ]["children"].items()
                        if child not in visited_states
                    },
                    path_generator.states_values,
                    path_generator.actions_values,
                    continuous=False,
                    grave=grave,
                    n_visits_reference=N_VISITS_REFERENCE,
                    reference_position=reference_position,
                )
            else:
                raise ValueError("Strategy in discrete MCTS ill-defined")
            if new_cell is None:
                no_cell_found = True
                break
            if path_generator.strategy in ["rave", "grave"]:
                path_generator.actions_values[tuple(path_generator.current_position)][normalized_angle][
                    "n_visits"
                ] += 1

            path_generator.update(normalized_angle, new_cell)
            assert (
                new_cell not in visited_states
            ), "Selected state was already visited: " + str(new_cell)
            visited_states.add(tuple(new_cell))
            selection_length += 1

        if no_cell_found:
            continue

        path_generator.states_values[tuple(path_generator.current_position)]["n_visits"] += 1

        # Stop iterating if all moves are selected
        if selection_length >= path_generator.trajectory_size:
            break

        # Expansion
        expansion = False
        if not path_generator.is_finished():
            # print("Expanding for iteration ", iteration_number + 1)
            if path_generator.strategy == "mcts":
                normalized_angle, new_cell = None, discrete_expansion(
                    path_generator.current_position, path_generator.states_values
                )
                path_generator.states_values[tuple(path_generator.current_position)]["children"].append(
                    new_cell
                )
                # print(new_cell, " vs ", path_generator.current_position)
                path_generator.states_values[tuple(path_generator.current_position)][
                    "unvisited_children"
                ].remove(new_cell)
            elif path_generator.strategy in ["rave", "grave"]:
                normalized_angle, new_cell = discrete_rave_expansion(
                    path_generator.current_position, path_generator.states_values
                )
                path_generator.states_values[tuple(path_generator.current_position)]["children"][
                    normalized_angle
                ] = new_cell
                # print(new_cell, " vs ", path_generator.current_position)
                path_generator.states_values[tuple(path_generator.current_position)][
                    "unvisited_children"
                ].pop(normalized_angle)
            else:
                raise ValueError("Strategy in discrete MCTS ill-defined")

            visited_states.add(tuple(new_cell))
            if len(visited_states) == path_generator.trajectory_size:
                print(visited_states)
            ## Add cell to visited states and remove it from unvisited ones wrt current position

            if new_cell in path_generator.states_values.keys():
                path_generator.states_values[tuple(new_cell)]["n_visits"] += 1

            else:
                path_generator.states_values[tuple(new_cell)] = {
                    "n_visits": 1,
                    "cumulative_score": 0,
                    "mean_score": 0,
                }
                if path_generator.strategy == "mcts":
                    path_generator.states_values[tuple(new_cell)]["children"] = list()
                    path_generator.states_values[tuple(new_cell)]["unvisited_children"] = (
                        list(
                            set(
                                list(
                                    discrete_possible_moves(
                                        new_cell, path_generator.current_map
                                    ).values()
                                )
                            )
                        )
                    )

                elif path_generator.strategy in ["rave", "grave"]:
                    path_generator.actions_values[tuple(new_cell)] = {
                        angle: {
                            "n_visits": 0,
                            "cumulative_score": 0,
                        }
                        for angle in DISCRETE_ACTIONS
                    }
                    path_generator.states_values[tuple(new_cell)]["children"] = dict()
                    path_generator.states_values[tuple(new_cell)]["unvisited_children"] = (
                        discrete_possible_moves(new_cell, path_generator.current_map)
                    )
                else:
                    raise ValueError("Strategy in discrete MCTS ill-defined")

            # print(path_generator.states_values[tuple(new_cell)]["unvisited_children"])
            path_generator.update(normalized_angle, new_cell)
            expansion = True

        # Simulation
        simulation_length = 0
        while not path_generator.is_finished():
            if path_generator.strategy == "mcts":
                new_cell = discrete_random_simulation(
                    path_generator.current_position, path_generator.current_map
                )
                normalized_angle = None
            elif path_generator.strategy in ["rave", "grave"]:
                normalized_angle, new_cell = discrete_rave_simulation(
                    path_generator.current_position, path_generator.current_map, path_generator.actions_values
                )
            else:
                raise ValueError("Strategy in discrete MCTS ill-defined")
            path_generator.update(normalized_angle, new_cell)
            simulation_length += 1

        # Backpropagation
        score = path_generator.get_score()
        backpropagation(
            path_generator.trajectory, path_generator.states_values, score / path_generator.score_normalizer
        )
        if path_generator.strategy in ["rave", "grave"]:
            rave_backpropagation(
                path_generator.actions, path_generator.actions_values, score / path_generator.score_normalizer
            )

        # if (iteration_number + 1) % 100 == 0:
            # print("At iteration ", iteration_number + 1)
            # print("Selection length: ", selection_length)
            # print("Expaned? ", expansion)
            # print("Simulation length: ", simulation_length, "\n")
            # if not expansion:
            #    print(visited_states)

        # Checking if a better score is found
        if score < best_score:
            best_trajectory = path_generator.trajectory[:]
            best_score = score
            print(
                "New best score found at iteration",
                iteration_number + 1,
                ": ",
                best_score,
            )

    path_generator.best_score = best_score
    path_generator.trajectory = best_trajectory

def cmcts(path_generator, n_iterations: int = 10000):
    """
    Continuous MCTS with Progressive Widening
    """
    best_trajectory = None
    best_score = path_generator.best_score

    for iteration_number in range(n_iterations):
        path_generator.reinitialize()
        visited_states = {code(path_generator.current_position)}
        reference_position = path_generator.start_point

        # Selection
        selection_length = 0
        no_cell_found = False
        while (not path_generator.is_finished()) and (
            path_generator.states_values[code(path_generator.current_position)]["n_visits"]
            ** (PROGRESSIVE_WIDENING_PARAMETER / (selection_length + 1))
            < len(path_generator.states_values[code(path_generator.current_position)]["children"])
        ):
            path_generator.states_values[code(path_generator.current_position)]["n_visits"] += 1
            if path_generator.strategy == "cmcts":
                new_cell = selection(
                    path_generator.current_position,
                    [
                        child
                        for child in path_generator.states_values[
                            code(path_generator.current_position)
                        ]["children"]
                        if child not in visited_states
                    ],
                    path_generator.states_values,
                )
                chosen_move = None
            elif path_generator.strategy in ["crave", "cgrave"]:
                chosen_move, new_cell = rave_selection(
                    path_generator.current_position,
                    {
                        move: child
                        for move, child in path_generator.states_values[
                            code(path_generator.current_position)
                        ]["children"].items()
                        if child not in visited_states
                    },
                    path_generator.states_values,
                    path_generator.actions_values,
                    continuous=True,
                    grave=(path_generator.strategy == "cgrave"),
                    n_visits_reference=N_VISITS_REFERENCE,
                    reference_position=reference_position,
                )
                chosen_move = code(chosen_move)
            else:
                raise ValueError("Strategy in discrete MCTS ill-defined")
            if new_cell is None:
                no_cell_found = True
                break
            # path_generator.actions_values[tuple(path_generator.current_position)][normalized_angle][
            #    "n_visits"
            # ] += 1
            path_generator.update(chosen_move, new_cell)
            assert (
                new_cell not in visited_states
            ), "Selected state was already visited: " + str(new_cell)
            visited_states.add(code(new_cell))
            selection_length += 1

        if no_cell_found:
            continue
        path_generator.states_values[code(path_generator.current_position)]["n_visits"] += 1
        # Stop iterating if all moves are selected
        if selection_length >= path_generator.trajectory_size:
            break

        # Expansion
        expansion = False
        if not path_generator.is_finished():
            move, new_cell = continuous_expansion(
                path_generator.current_position, path_generator.states_values, path_generator.current_map
            )
            move, new_cell = code(move), code(new_cell)
            if code(new_cell) in path_generator.states_values.keys():
                path_generator.states_values[code(new_cell)]["n_visits"] += 1

            else:
                path_generator.states_values[code(new_cell)] = {
                    "n_visits": 1,
                    "cumulative_score": 0,
                    "mean_score": 0,
                    "children": list(),
                }
                if path_generator.strategy in ["crave", "cgrave"]:
                    path_generator.states_values[code(new_cell)]["children"] = dict()
            if path_generator.strategy == "cmcts":
                path_generator.states_values[code(path_generator.current_position)]["children"].append(
                    new_cell
                )
            elif path_generator.strategy in ["crave", "cgrave"]:
                path_generator.states_values[code(path_generator.current_position)]["children"][
                    move
                ] = new_cell
            else:
                raise ValueError("Strategy in discrete MCTS ill-defined")

            path_generator.update(move, new_cell)
            expansion = True

        # Simulation
        simulation_length = 0
        while not path_generator.is_finished():
            move, new_cell = continuous_random_simulation(
                path_generator.current_position, path_generator.current_map
            )
            path_generator.update(code(move), code(new_cell))
            simulation_length += 1

        score = path_generator.get_score()
        backpropagation(
            path_generator.trajectory, path_generator.states_values, score / path_generator.score_normalizer
        )

        # if (iteration_number + 1) % 100 == 0:
        #     print("At iteration ", iteration_number + 1)
        #     print("Selection length: ", selection_length)
        #     print("Expaned? ", expansion)
        #     print("Simulation length: ", simulation_length, "\n")

        # Checking if a better score is found
        if score < best_score:
            best_trajectory = path_generator.trajectory[:]
            best_score = score
            # print(
            #     "New best score found at iteration",
            #     iteration_number + 1,
            #     ": ",
            #     best_score,
            #     "\n",
            # )

    path_generator.best_score = best_score
    path_generator.trajectory = best_trajectory