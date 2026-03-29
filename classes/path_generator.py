import random
from copy import deepcopy
import numpy as np
from solvers.nrpa import *
from solvers.gnrpa import *
from solvers.abgnrpa import *
from solvers.mcts import *
from solvers.rave import *
from utils.constants import *
from utils.sampling_utils import *
from utils.basic_functions import *
from utils.map_utils import *
from classes.heuristic_values import HeuristicValues


class PathGenerator(object):

    def __init__(
        self,
        current_map,
        start_point,
        goal,
        trajectory_size,
        strategy,
        bias_factor=1,
    ):
        self.current_map = current_map
        self.start_point = start_point
        self.goal = goal
        self.trajectory_size = trajectory_size
        self.strategy = strategy
        self.current_position = start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]
        self.actions = list()
        self.best_course_of_actions = list()
        height, width = current_map.shape
        max_radius = np.sqrt(height**2 + width**2)
        self.relevance_levels = [0.5]
        radius = 0.5
        while radius < max_radius:
            radius *= 2
            self.relevance_levels.append(radius)
        self.best_score = height * width
        self.score_normalizer = height * width
        if strategy in ["nrpa", "gnrpa", "abgnrpa"]:
            self.policy = dict()
            self.nrpa_iterations = 0
            self.heuristic_values = HeuristicValues(bias_factor)
        if strategy in ["mcts", "rave", "grave"]:
            self.states_values = {
                tuple(start_point): {
                    "n_visits": 0,
                    "cumulative_score": 0,
                    "mean_score": 0,
                    "children": list(),
                    "unvisited_children": list(),
                }
            }
            if strategy in ["rave", "grave"]:
                self.reference_state = start_point
                self.states_values[tuple(start_point)]["unvisited_children"] = dict()
                self.states_values[tuple(start_point)]["children"] = dict()
                self.actions_values = {
                    tuple(start_point): {
                        angle: {
                            "n_visits": 0,
                            "cumulative_score": 0,
                        }
                        for angle in DISCRETE_ACTIONS
                    }
                }
        elif strategy in ["cmcts", "crave", "cgrave"]:
            self.states_values = {
                tuple(start_point): {
                    "n_visits": 1,
                    "cumulative_score": 0,
                    "mean_score": 0,
                    "children": list(),
                }
            }
            if strategy in ["crave", "cgrave"]:
                self.reference_state = start_point
                self.states_values[tuple(start_point)]["children"] = dict()
                self.actions_values = {tuple(start_point): dict()}
        elif strategy == "cnmcts":
            self.states_actions = dict()

    def is_finished(self):
        return (
                np.linalg.norm(
                np.array(list(self.current_position)) - np.array(list(self.goal))
            ) < 0.5
        ) or (self.current_steps >= self.trajectory_size)

    def reinitialize(self):
        self.current_position = self.start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]
        self.actions = list()
        if self.strategy in ["mcts", "rave", "grave", "crave", "cgrave"]:
            self.reference_state = self.start_point

    def get_score(self):
        return len(self.trajectory) + np.linalg.norm(
            np.array(self.goal) - np.array(self.current_position)
        )

    def update(self, move, new_cell):
        self.current_position = code(new_cell)
        self.trajectory.append(code(new_cell))
        self.actions.append(code(move))
        self.current_steps += 1

    def step(self):
        self.current_position = code(self.current_position)
        if self.strategy in ["nrpa", "gnrpa", "abgnrpa"]:
            sampling_radius = np.exp(
                -self.nrpa_iterations / (self.n_policies * HALF_LIFE_DIVIDER)
            )  # * np.sqrt(self.current_steps / self.trajectory_size)
        if self.strategy == "random_walk":
            return continuous_random_simulation(self.current_position, self.current_map)

        elif self.strategy == "nrpa":
            move = nrpa_step(
                self.current_position,
                self.current_map,
                self.policy,
                sampling_radius=sampling_radius,
                relevance_radius_list=self.relevance_levels,
            )

        elif self.strategy == "gnrpa":
            move = gnrpa_step(
                self.current_position,
                self.goal,
                self.current_map,
                self.policy,
                sampling_radius=sampling_radius,
                relevance_radius_list=self.relevance_levels,
            )

        elif self.strategy == "abgnrpa":
            move = abgnrpa_step(
                self.current_position,
                self.goal,
                self.current_map,
                self.policy,
                self.heuristic_values,
                1 - self.get_score() / self.score_normalizer,
                sampling_radius=sampling_radius,
                relevance_radius_list=self.relevance_levels,
            )

        else:
            raise (ValueError("No valid strategy defined"))

        return code(move), continuous_cell_selector(self.current_position, move)

    def generate_path(self):

        self.trajectory = [self.start_point]
        while not self.is_finished():
            move, new_cell = self.step()
            assert cell_is_reachable(
                self.current_position, new_cell, self.current_map
            ), ("Position " + str(new_cell) + " out of bounds OR inside obstacle")
            self.update(move, new_cell)

    def adapt_policy(
        self,
        best_trajectory,
        policy,
        learning_rate,
    ):
        if self.strategy == "nrpa":
            return adapt_policy_nrpa(
                best_trajectory,
                policy,
                learning_rate,
            )
        elif self.strategy in ["gnrpa", "abgnrpa"]:
            return adapt_policy_gnrpa(
                best_trajectory,
                policy,
                learning_rate,
            )
        else:
            raise ValueError("Wrong strategy for policy adaptation")

    def nrpa(self, level: int = 1, n_policies: int = 100):
        self.n_policies = n_policies
        trajectory_evolution, score_evolution = list(), list()
        if level == 0:
            self.reinitialize()
            self.nrpa_iterations += 1
            self.generate_path()
            score_evolution.append(self.get_score())
            return score_evolution, list()

        else:
            iteration_number = self.nrpa_iterations
            best_score = self.best_score
            last_best_score = best_score
            best_trajectory = deepcopy(self.trajectory)
            best_course_of_actions = deepcopy(self.actions)
            policy = deepcopy(self.policy)
            learning_rate = np.sqrt(1 / self.trajectory_size)
            self.cumulative_change = 100
            for iteration_number in range(n_policies):
                score_list, trajectory_list = self.nrpa(level - 1, n_policies)
                score_evolution.extend(score_list)
                trajectory_evolution.extend(trajectory_list)
                score = self.get_score()
                if score < best_score:
                    best_score, score = score, best_score
                    best_trajectory = deepcopy(self.trajectory)
                    best_course_of_actions = deepcopy(self.actions)
                    trajectory_evolution.append(self.get_trajectory_frame())
                    print(
                        "Better score found at iteration ",
                        iteration_number + 1,
                        ": ",
                        int(best_score),
                    )
                self.policy = self.adapt_policy(
                    best_trajectory,
                    self.policy,
                    learning_rate,
                )
                score = self.get_score()
                if (iteration_number + 1) % 100 == 0:
                    print(
                        "Iteration n° ",
                        iteration_number + 1,
                        ": best score: ",
                        best_score,
                    )
                score_evolution.append(score)
            self.nrpa_iterations = iteration_number + 1
            policy = self.adapt_policy(
                best_trajectory,
                policy=policy,
                learning_rate=learning_rate,
            )
            self.policy = deepcopy(policy)
            self.trajectory = deepcopy(best_trajectory)
            self.best_score = best_score

            # Plotting score_evolution
            return score_evolution, trajectory_evolution

    def mcts(self, n_iterations: int = 10000):
        """
        Discrete MCTS
        """
        best_trajectory = None
        best_score = self.best_score
        if self.strategy == "mcts":
            self.states_values[tuple(self.current_position)]["unvisited_children"] = (
                list(
                    discrete_possible_moves(
                        self.current_position, self.current_map
                    ).values()
                )
            )
        elif self.strategy in ["rave", "grave"]:
            self.states_values[tuple(self.current_position)]["unvisited_children"] = (
                discrete_possible_moves(self.current_position, self.current_map)
            )
        (
            trajectory_evolution,
            score_evolution,
            score_variation,
            selection_length_list,
        ) = (list(), list(), list(), list())
        # print(len(self.states_values[tuple(self.current_position)]["unvisited_children"]))
        for iteration_number in range(n_iterations):
            self.reinitialize()
            visited_states = {tuple(self.start_point)}
            reference_position = self.start_point

            # Selection
            selection_length = 0
            no_cell_found = False
            while (
                len(
                    self.states_values[tuple(self.current_position)][
                        "unvisited_children"
                    ]
                )
                == 0
            ) and not self.is_finished():
                self.states_values[tuple(self.current_position)]["n_visits"] += 1
                if self.strategy == "mcts":
                    new_cell = selection(
                        self.current_position,
                        [
                            child
                            for child in self.states_values[
                                tuple(self.current_position)
                            ]["children"]
                            if child not in visited_states
                        ],
                        self.states_values,
                    )
                    normalized_angle = None
                elif self.strategy in ["rave", "grave"]:
                    grave = False
                    if self.strategy == "grave":
                        grave = True
                    normalized_angle, new_cell = rave_selection(
                        self.current_position,
                        {
                            angle: child
                            for angle, child in self.states_values[
                                tuple(self.current_position)
                            ]["children"].items()
                            if child not in visited_states
                        },
                        self.states_values,
                        self.actions_values,
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
                if self.strategy in ["rave", "grave"]:
                    self.actions_values[tuple(self.current_position)][normalized_angle][
                        "n_visits"
                    ] += 1

                self.update(normalized_angle, new_cell)
                assert (
                    new_cell not in visited_states
                ), "Selected state was already visited: " + str(new_cell)
                visited_states.add(tuple(new_cell))
                selection_length += 1

            if no_cell_found:
                continue

            self.states_values[tuple(self.current_position)]["n_visits"] += 1
            selection_length_list.append(selection_length)

            # Stop iterating if all moves are selected
            if selection_length >= self.trajectory_size:
                break

            # Expansion
            expansion = False
            if not self.is_finished():
                # print("Expanding for iteration ", iteration_number + 1)
                if self.strategy == "mcts":
                    normalized_angle, new_cell = None, discrete_expansion(
                        self.current_position, self.states_values
                    )
                    self.states_values[tuple(self.current_position)]["children"].append(
                        new_cell
                    )
                    # print(new_cell, " vs ", self.current_position)
                    self.states_values[tuple(self.current_position)][
                        "unvisited_children"
                    ].remove(new_cell)
                elif self.strategy in ["rave", "grave"]:
                    normalized_angle, new_cell = discrete_rave_expansion(
                        self.current_position, self.states_values
                    )
                    self.states_values[tuple(self.current_position)]["children"][
                        normalized_angle
                    ] = new_cell
                    # print(new_cell, " vs ", self.current_position)
                    self.states_values[tuple(self.current_position)][
                        "unvisited_children"
                    ].pop(normalized_angle)
                else:
                    raise ValueError("Strategy in discrete MCTS ill-defined")

                visited_states.add(tuple(new_cell))
                if len(visited_states) == self.trajectory_size:
                    print(visited_states)
                ## Add cell to visited states and remove it from unvisited ones wrt current position

                if new_cell in self.states_values.keys():
                    self.states_values[tuple(new_cell)]["n_visits"] += 1

                else:
                    self.states_values[tuple(new_cell)] = {
                        "n_visits": 1,
                        "cumulative_score": 0,
                        "mean_score": 0,
                    }
                    if self.strategy == "mcts":
                        self.states_values[tuple(new_cell)]["children"] = list()
                        self.states_values[tuple(new_cell)]["unvisited_children"] = (
                            list(
                                set(
                                    list(
                                        discrete_possible_moves(
                                            new_cell, self.current_map
                                        ).values()
                                    )
                                )
                            )
                        )

                    elif self.strategy in ["rave", "grave"]:
                        self.actions_values[tuple(new_cell)] = {
                            angle: {
                                "n_visits": 0,
                                "cumulative_score": 0,
                            }
                            for angle in DISCRETE_ACTIONS
                        }
                        self.states_values[tuple(new_cell)]["children"] = dict()
                        self.states_values[tuple(new_cell)]["unvisited_children"] = (
                            discrete_possible_moves(new_cell, self.current_map)
                        )
                    else:
                        raise ValueError("Strategy in discrete MCTS ill-defined")

                # print(self.states_values[tuple(new_cell)]["unvisited_children"])
                self.update(normalized_angle, new_cell)
                expansion = True

            # Simulation
            simulation_length = 0
            while not self.is_finished():
                if self.strategy == "mcts":
                    new_cell = discrete_random_simulation(
                        self.current_position, self.current_map
                    )
                    normalized_angle = None
                elif self.strategy in ["rave", "grave"]:
                    normalized_angle, new_cell = discrete_rave_simulation(
                        self.current_position, self.current_map, self.actions_values
                    )
                else:
                    raise ValueError("Strategy in discrete MCTS ill-defined")
                self.update(normalized_angle, new_cell)
                simulation_length += 1

            # Backpropagation
            score = self.get_score()
            backpropagation(
                self.trajectory, self.states_values, score / self.score_normalizer
            )
            if self.strategy in ["rave", "grave"]:
                rave_backpropagation(
                    self.actions, self.actions_values, score / self.score_normalizer
                )

            if (iteration_number + 1) % 100 == 0:
                print("At iteration ", iteration_number + 1)
                print("Selection length: ", selection_length)
                print("Expaned? ", expansion)
                print("Simulation length: ", simulation_length, "\n")
                # if not expansion:
                #    print(visited_states)

            # Checking if a better score is found
            if score < best_score:
                best_trajectory = self.trajectory[:]
                best_score = score
                print(
                    "New best score found at iteration",
                    iteration_number + 1,
                    ": ",
                    best_score,
                )
                trajectory_evolution.append(self.get_trajectory_frame())
            score_evolution.append(best_score)
            score_variation.append(score)

        self.best_score = score_evolution[-1]
        self.trajectory = best_trajectory

        # Plotting score_evolution
        figure, axe = plt.subplots(2)
        timer = figure.canvas.new_timer(interval=5000)
        timer.add_callback(plt.close)
        timer.start()
        axe[0].plot(score_evolution, color="r", label="Best score evolution")
        axe[0].plot(score_variation, color="b", label="Score variation")
        axe[1].plot(selection_length_list, label="Selection length evolution")
        axe[0].legend()
        axe[1].legend()
        plt.show()
        plt.close()
        play_scenario(
            trajectory_evolution, self.strategy.upper(), best_score, wait_time=1
        )

    def cmcts(self, n_iterations: int = 10000):
        """
        Continuous MCTS with Progressive Widening
        """
        best_trajectory = None
        best_score = self.best_score
        (
            trajectory_evolution,
            score_evolution,
            score_variation,
            selection_length_list,
        ) = (list(), list(), list(), list())

        for iteration_number in range(n_iterations):
            self.reinitialize()
            visited_states = {tuple(self.current_position)}
            reference_position = self.start_point

            # Selection
            selection_length = 0
            no_cell_found = False
            while (not self.is_finished()) and (
                self.states_values[tuple(self.current_position)]["n_visits"]
                ** (PROGRESSIVE_WIDENING_PARAMETER / (selection_length + 1))
                < len(self.states_values[tuple(self.current_position)]["children"])
            ):
                self.states_values[tuple(self.current_position)]["n_visits"] += 1
                if self.strategy == "cmcts":
                    new_cell = selection(
                        self.current_position,
                        [
                            child
                            for child in self.states_values[
                                tuple(self.current_position)
                            ]["children"]
                            if child not in visited_states
                        ],
                        self.states_values,
                    )
                    normalized_angle = None
                elif self.strategy in ["crave", "cgrave"]:
                    normalized_angle, new_cell = rave_selection(
                        self.current_position,
                        {
                            angle: child
                            for angle, child in self.states_values[
                                tuple(self.current_position)
                            ]["children"].items()
                            if child not in visited_states
                        },
                        self.states_values,
                        self.actions_values,
                        continuous=True,
                        grave=(self.strategy == "cgrave"),
                        n_visits_reference=N_VISITS_REFERENCE,
                        reference_position=reference_position,
                    )
                    normalized_angle = code(normalized_angle)
                else:
                    raise ValueError("Strategy in discrete MCTS ill-defined")
                if new_cell is None:
                    no_cell_found = True
                    break
                # self.actions_values[tuple(self.current_position)][normalized_angle][
                #    "n_visits"
                # ] += 1
                self.update(normalized_angle, new_cell)
                assert (
                    new_cell not in visited_states
                ), "Selected state was already visited: " + str(new_cell)
                visited_states.add(tuple(new_cell))
                selection_length += 1

            if no_cell_found:
                continue
            self.states_values[tuple(self.current_position)]["n_visits"] += 1
            selection_length_list.append(selection_length)
            # Stop iterating if all moves are selected
            if selection_length >= self.trajectory_size:
                break

            # Expansion
            expansion = False
            if not self.is_finished():
                normalized_angle, new_cell = continuous_expansion(
                    self.current_position, self.states_values, self.current_map
                )
                normalized_angle, new_cell = code(normalized_angle), code(new_cell)
                if tuple(new_cell) in self.states_values.keys():
                    self.states_values[tuple(new_cell)]["n_visits"] += 1

                else:
                    self.states_values[tuple(new_cell)] = {
                        "n_visits": 1,
                        "cumulative_score": 0,
                        "mean_score": 0,
                        "children": list(),
                    }
                    if self.strategy in ["crave", "cgrave"]:
                        self.states_values[tuple(new_cell)]["children"] = dict()
                if self.strategy == "cmcts":
                    self.states_values[tuple(self.current_position)]["children"].append(
                        new_cell
                    )
                elif self.strategy in ["crave", "cgrave"]:
                    self.states_values[tuple(self.current_position)]["children"][
                        normalized_angle
                    ] = new_cell
                else:
                    raise ValueError("Strategy in discrete MCTS ill-defined")

                self.update(normalized_angle, new_cell)
                expansion = True

            # Simulation
            simulation_length = 0
            while not self.is_finished():
                normalized_angle, new_cell = continuous_random_simulation(
                    self.current_position, self.current_map
                )
                self.update(code(normalized_angle), code(new_cell))
                simulation_length += 1

            score = self.get_score()
            backpropagation(
                self.trajectory, self.states_values, score / self.score_normalizer
            )

            if (iteration_number + 1) % 100 == 0:
                print("At iteration ", iteration_number + 1)
                print("Selection length: ", selection_length)
                print("Expaned? ", expansion)
                print("Simulation length: ", simulation_length, "\n")

            # Checking if a better score is found
            if score < best_score:
                best_trajectory = self.trajectory[:]
                best_score = score
                print(
                    "New best score found at iteration",
                    iteration_number + 1,
                    ": ",
                    best_score,
                    "\n",
                )
                trajectory_evolution.append(self.get_trajectory_frame())
            score_evolution.append(best_score)
            score_variation.append(score)

        self.best_score = score_evolution[-1]
        self.trajectory = best_trajectory

        # Plotting score_evolution
        figure, axe = plt.subplots(2)
        timer = figure.canvas.new_timer(interval=5000)
        timer.add_callback(plt.close)
        timer.start()
        axe[0].plot(score_evolution, color="r", label="Best score evolution")
        axe[0].plot(score_variation, color="b", label="Score variation")
        axe[1].plot(selection_length_list, label="Selection length evolution")
        axe[0].legend()
        axe[1].legend()
        plt.show()
        plt.close()
        play_scenario(
            trajectory_evolution, self.strategy.upper(), best_score, wait_time=1
        )

    def cnmcts(
        self, level: int = 1, bandwidth: int = 50
    ):
        if level == 0:
            while not self.is_finished():
                if code(self.current_position) in self.states_actions.keys():
                    move = random.choice(
                        self.states_actions[code(self.current_position)]
                    )
                    new_cell = continuous_cell_selector(self.current_position, move)
                else:
                    move, new_cell = continuous_random_simulation(
                        self.current_position, self.current_map
                    )
                self.update(move, new_cell)
            return self.trajectory, self.actions, self.get_score()

        else:
            while not self.is_finished():
                # scores_list = list()
                if not code(self.current_position) in self.states_actions.keys():
                    self.states_actions[code(self.current_position)] = list()
                    for _ in range(bandwidth):
                        new_cell = [-1, -1]
                        # visited_state = False
                        while not cell_is_reachable(
                            self.current_position, new_cell, self.current_map
                        ):
                            move, new_cell = continuous_random_simulation(
                                self.current_position, self.current_map
                            )
                        self.states_actions[code(self.current_position)].append(move)
                moves_list = self.states_actions[code(self.current_position)]
                for move_index, move in enumerate(moves_list):
                    move_scores_list = list()
                    # for _ in range(node_iterations):
                    path = deepcopy(self)
                    path.update(
                        move, continuous_cell_selector(path.current_position, move)
                    )
                    trajectory, actions_list, score = path.cnmcts(
                        level - 1, bandwidth
                    )
                    if score < self.best_score:
                        self.best_trajectory = trajectory[:]
                        self.best_course_of_actions = actions_list[:]
                        self.best_score = score
                # TODO: investigate the reason behind the need for the next if clause
                # It is not supposed to be needed
                if self.current_steps >= len(self.best_course_of_actions):
                    self.actions = self.best_course_of_actions[:][:-1]
                    self.trajectory = self.best_trajectory[:][:-1]
                    self.current_steps = len(self.best_course_of_actions) - 1
                self.update(
                    self.best_course_of_actions[self.current_steps],
                    self.best_trajectory[self.current_steps + 1],
                )

        return self.best_trajectory, self.best_course_of_actions, self.best_score

    def policy_by_region_nrpa(self, level: int, n_policies: int):
        pass

    def get_movement_frames(self):
        frames = [get_map(self.current_map, [self.start_point], self.goal)]
        passage_points = list()
        for cell_number, cell in enumerate(self.trajectory[:-1]):
            passage_points.extend(
                get_intermediary_passage_points(
                    cell, self.trajectory[cell_number + 1], self.current_map
                )
            )
            frames.append(get_map(self.current_map, passage_points, self.goal))

        return frames

    def get_trajectory_frame(self):
        passage_points = list()
        for cell_number, cell in enumerate(self.trajectory[:-1]):
            passage_points.extend(
                get_intermediary_passage_points(
                    (int(cell[0]), int(cell[1])),
                    (
                        int(self.trajectory[cell_number + 1][0]),
                        int(self.trajectory[cell_number + 1][1]),
                    ),
                    self.current_map,
                )
            )
        if len(passage_points) == 0:
            print("Trajectory length: ", len(self.trajectory))
            print("Passage points: ", len(passage_points))
            print("Score: ", self.get_score())
            print("Current position: ", self.current_position)
            print("Goal: ", self.goal)
            print("N° of steps: ", self.current_steps)
        return get_map(self.current_map, passage_points, self.goal)

    def run(self, inputs: dict):
        if self.strategy == "random_walk":
            trajectories, scores = list(), list()
            for _ in range(inputs["n_iterations"]):
                self.reinitialize()
                self.generate_path()
                trajectories.append(self.trajectory[:])
                scores.append(self.get_score())
            best_score_index = np.argmin(scores)
            self.trajectory = trajectories[best_score_index]
            self.best_score = scores[best_score_index]
        elif self.strategy in ["nrpa", "gnrpa", "abgnrpa"]:
            self.nrpa(level=inputs["level"], n_policies=inputs["n_iterations"])
        elif self.strategy in ["mcts", "rave", "grave"]:
            self.mcts(n_iterations=inputs["n_iterations"])
        elif self.strategy in ["cmcts", "crave", "cgrave"]:
            self.cmcts(n_iterations=inputs["n_iterations"])
        elif self.strategy == "cnmcts":
            trajectory, actions, score = self.cnmcts(
                level=inputs["level"],
                bandwidth=inputs["bandwidth"],
            )
            self.trajectory = trajectory
            self.actions = actions
            self.best_score = score
