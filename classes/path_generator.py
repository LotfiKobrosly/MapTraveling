import random
from copy import deepcopy
import numpy as np
from solvers.nrpa import *
from solvers.gnrpa import *
from solvers.abgnrpa import *
from solvers.mcts import *
from solvers.crave import compute_rave_uct, compute_grave_uct
from utils.constants import *
from utils.sampling_utils import *
from utils.map_utils import *
from classes.heuristic_values import HeuristicValues


class PathGenerator(object):

    def __init__(self, current_map, start_point, goal, trajectory_size, strategy):
        self.current_map = current_map
        self.start_point = start_point
        self.goal = goal
        self.trajectory_size = trajectory_size
        self.strategy = strategy
        self.current_position = start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]
        self.actions = list()
        height, width = current_map.shape
        self.best_score = height * width
        if strategy in ["nrpa", "gnrpa", "abgnrpa"]:
            self.policy = dict()
            self.nrpa_iterations = 0
            self.heuristic_values = HeuristicValues(bias_factor=1)
        if strategy in ["mcts", "crave", "cgrave"]:
            self.states_values = {
                tuple(start_point): {
                    "n_visits": 0,
                    "cumulative_score": 0,
                    "mean_score": 0,
                    "children": list(),
                }
            }
            self.expanded = False
            self.reference_state = start_point

    def is_finished(self):
        return (
            (self.goal[0] == self.current_position[0])
            and (self.goal[1] == self.current_position[1])
            or (self.current_steps == self.trajectory_size)
        )

    def reinitialize(self):
        self.current_position = self.start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]
        if self.strategy in ["mcts", "crave", "cgrave"]:
            self.expanded = False
            self.reference_state = self.start_point

    def step(self):
        if self.strategy == "random_walk":
            height, width = self.current_map.shape
            return random_simulation(self.current_position, self.current_map)

        elif self.strategy == "nrpa":
            normalized_angle = nrpa_step(self.current_position, self.policy)

        elif self.strategy == "gnrpa":
            normalized_angle = gnrpa_step(self.current_position, self.goal, self.policy)

        elif self.strategy == "abgnrpa":
            normalized_angle = abgnrpa_step(
                self.current_position,
                self.goal,
                self.policy,
                self.heuristic_values,
                self.get_score(),
            )

        else:
            raise (ValueError("No strategy defined"))

        return normalized_angle, cell_selector(
            self.current_position, normalized_angle * 2 * np.pi
        )

    def update(self, angle, new_cell):
        self.current_steps += 1
        self.current_position = new_cell
        self.trajectory.append(new_cell)
        self.actions.append(angle)

    def adapt_policy(self, best_trajectory, best_course_of_actions, policy, best_score):
        if self.strategy == "nrpa":
            return adapt_policy_nrpa(
                best_trajectory, best_course_of_actions, policy, best_score
            )
        elif self.strategy in ["gnrpa", "abgnrpa"]:
            return adapt_policy_gnrpa(
                best_trajectory, best_course_of_actions, policy, best_score
            )
        else:
            raise ValueError("Wrong strategy for policy adaptation")

    def nrpa(self, level: int = 1, n_policies: int = 100):

        if level == 0:
            self.generate_path()
            self.nrpa_iterations += 1

        else:
            iteration_number = self.nrpa_iterations
            best_score = self.best_score
            best_trajectory = deepcopy(self.trajectory)
            best_course_of_actions = deepcopy(self.actions)
            policy = deepcopy(self.policy)
            for _ in range(n_policies):
                self.reinitialize()
                self.nrpa(level - 1, n_policies)
                score = self.get_score()
                if score < best_score:
                    best_score = score
                    best_trajectory = deepcopy(self.trajectory)
                    best_course_of_actions = deepcopy(self.actions)
                self.policy = self.adapt_policy(
                    best_trajectory, best_course_of_actions, self.policy, best_score
                )
            self.nrpa_iterations = iteration_number + 1
            policy = self.adapt_policy(
                best_trajectory,
                best_course_of_actions,
                policy=policy,
                best_score=best_score,
            )
            self.policy = policy
            self.trajectory = best_trajectory
            self.best_score = best_score

    def mcts(self, n_iterations: int = 10000):
        """
        Continuous MCTS with Progressive Widening
        """
        best_trajectory = None
        best_score = self.best_score
        for iteration_number in range(n_iterations):
            self.reinitialize()
            while not self.is_finished():
                if self.expanded:
                    angle, new_cell = random_simulation(self.current_position, self.current_map)
                
                else:
                    current_node = self.states_values[tuple(self.current_position)]
                    if current_node["n_visits"] ** PROGRESSIVE_WIDENING_PARAMETER >= len(current_node["children"]):
                        angle, new_cell = random_simulation(self.current_position, self.current_map)
                        current_node["children"].append(new_cell)
                        if self.states_values.get(tuple(new_cell), None) is None:
                            self.states_values[tuple(new_cell)] = {
                                "n_visits": 1,
                                "cumulative_score": 0,
                                "mean_score": 0,
                                "children": list(),
                            }
                        else:
                            self.states_values[tuple(new_cell)]["n_visits"] += 1
                        self.expanded = True
                    else:
                        current_node["n_visits"] += 1
                        probabilites = np.zeros(len(current_node["children"]))
                        for (child_id, child) in enumerate(current_node["children"]):
                            #print(child_id, child)
                            if self.strategy == "mcts":
                                probabilites[child_id] = compute_uct(self.current_position, child, self.states_values)
                            elif self.strategy == "crave":
                                probabilites[child_id] = compute_rave_uct(self.current_position, child, self.states_values)
                            elif self.strategy == "cgrave":
                                if current_node["n_visits"] > N_VISITS_REFERENCE:
                                    self.reference_state = self.current_position
                                probabilites[child_id] = compute_grave_uct(self.current_position, child, self.reference_state, self.states_values)
                            else:
                                raise ValueError("Wrong strategy chosen: " + self.strategy.upper())
                            if not np.isfinite(probabilites[child_id]):
                                probabilites[child_id] = 1
                        angle = None
                        #print(probabilites)
                        probabilites /= np.sum(probabilites)
                        candidate_ids = np.arange(len(current_node["children"]))
                        new_cell = current_node["children"][np.random.choice(candidate_ids, size=1, p=probabilites)[0]]
                        
                self.update(angle, new_cell)
            score = self.get_score()
            backpropagation(self.trajectory, self.states_values, score, iteration_number)

            if score < best_score:
                best_trajectory = self.trajectory[:]
                best_score = score

        self.best_score = best_score
        self.trajectory = best_trajectory



    def generate_path(self):

        self.trajectory = [self.start_point]
        while not self.is_finished():
            angle, new_cell = random_simulation(self.current_position, self.current_map)
            self.update(angle, new_cell)

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

    def get_score(self):
        return len(self.trajectory) + np.linalg.norm(
            np.array(self.goal) - np.array(self.current_position)
        )

    def run(self, inputs: dict):
        if self.strategy == "random_walk":
            trajectories, scores = list(), list()
            for _ in range(inputs["n_iterations"]):
                self.generate_path()
                trajectories.append(self.trajectory[:])
                scores.append(self.get_score())
                self.reinitialize()
            best_score_index = np.argmin(scores)
            self.trajectory = trajectories[best_score_index]
            self.best_score = scores[best_score_index]
        elif self.strategy in ["nrpa", "gnrpa", "abgnrpa"]:
            self.nrpa(level=inputs["level"], n_policies=inputs["n_iterations"])
        elif self.strategy in ["mcts", "crave", "cgrave"]:
            self.mcts(n_iterations=inputs["n_iterations"])
