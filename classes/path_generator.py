import random
from copy import deepcopy
import numpy as np
from solvers.nrpa import *
from solvers.gnrpa import *
from utils.constants import *
from utils.sampling_utils import *
from utils.map_utils import *


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

    def step(self):
        if self.strategy == "random_walk":
            normalized_angle = RANDOM_STATE.uniform(0, 1)

        elif self.strategy == "nrpa":
            normalized_angle = nrpa_step(self.current_position, self.policy)

        elif self.strategy == "gnrpa":
            normalized_angle = gnrpa_step(self.current_position, self.goal, self.policy)

        else:
            raise (ValueError("No strategy defined"))

        return normalized_angle, cell_selector(
            self.current_position, normalized_angle * 2 * np.pi
        )

    def adapt_policy(self, best_trajectory, best_course_of_actions, policy, best_score):
        if self.strategy == "nrpa":
            return adapt_policy_nrpa(best_trajectory, best_course_of_actions, policy, best_score)
        elif self.strategy == "gnrpa":
            return adapt_policy_gnrpa(best_trajectory, best_course_of_actions, policy, best_score)
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

    def generate_path(self):
        height, width = self.current_map.shape

        self.trajectory = [self.start_point]
        while not self.is_finished():
            next_cell_found = False
            while not next_cell_found:
                angle, new_cell = self.step()
                if (
                    (new_cell[0] >= 0)
                    and (new_cell[0] < height)
                    and (new_cell[1] >= 0)
                    and (new_cell[1] < width)
                    and (self.current_map[new_cell[0], new_cell[1]] != 1)
                ):
                    next_cell_found = True
                    intermediary_passage_points = get_intermediary_passage_points(
                        self.current_position, new_cell, self.current_map
                    )
                    for point in intermediary_passage_points:
                        if self.current_map[point[0], point[1]] == 1:
                            next_cell_found = False
                            break
                    if not next_cell_found:
                        continue
                    break

            self.current_steps += 1
            self.current_position = new_cell
            self.trajectory.append(new_cell)
            self.actions.append(angle)

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
        elif self.strategy in ["nrpa", "gnrpa"]:
            self.nrpa(level=inputs["level"], n_policies=inputs["n_iterations"])