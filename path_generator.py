import random
from copy import deepcopy
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import *

RANDOM_STATE = np.random.RandomState(seed=42)
SAMPLE_SIZE = 1000


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
        if strategy == "nrpa":
            self.policy = [
                [
                    dict() for j in range(width)
                ]
                for i in range(height)
            ]
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

    def code_action(self, angle: float):
        return round(angle, 3)

    def step(self):
        if self.strategy == "random_walk":
            normalized_angle = random.random()
            

        elif self.strategy == "nrpa":
            position = self.current_position
            policy = self.policy[position[0]][position[1]]
            if policy:
                mixture_data = np.concatenate(
                    [
                        RANDOM_STATE.normal(key, value, SAMPLE_SIZE)
                        for (key, value) in policy.items()
                    ]
                ).reshape(-1, 1)
                normalized_angle = GaussianMixture(1).fit(mixture_data).sample(1)[0][0]
            else:
                normalized_angle = random.random()

        else:
            raise(ValueError("No strategy defined"))

        return normalized_angle, cell_selector(self.current_position, normalized_angle * 2 * np.pi)

    def adapt_policy(self, best_trajectory, best_course_of_actions, policy=None):
        if policy is None:
            policy = self.policy
        for (point_index, point) in enumerate(best_trajectory[:-1]):
            i, j = point[0], point[1]
            current_policy = policy[i][j]
            if current_policy:
                average_std = np.mean(np.array(list(current_policy.values())))
                for (key, value) in current_policy.items():
                    current_policy[key] = value * 10 / 9
                current_policy[self.code_action(best_course_of_actions[point_index])] = average_std * 9 / 10
            else:
                current_policy[self.code_action(best_course_of_actions[point_index])] = 0.5

    def nrpa(self, level: int=1, n_policies: int=100):

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
                self.adapt_policy(best_trajectory, best_course_of_actions)
            self.nrpa_iterations = iteration_number + 1
            self.adapt_policy(best_trajectory, best_course_of_actions, policy=policy)
            self.policy = policy
            self.trajectory = best_trajectory
            self.best_score = best_score

    def generate_path(self):
        height, width = self.current_map.shape
        
        self.trajectory = [self.start_point]
        passage_points = [self.start_point]
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
                    intermediary_passage_points = get_intermediary_passage_points(self.current_position, new_cell, self.current_map)
                    for point in intermediary_passage_points:
                        if self.current_map[point[0], point[1]] == 1:
                            next_cell_found = False
                            break
                    if not next_cell_found:
                        continue
                    passage_points.extend(intermediary_passage_points)
                    break
            self.current_steps += 1
            self.current_position = new_cell
            self.trajectory.append(new_cell)
            self.actions.append(angle)
            
    def get_movement_frames(self):
        frames = [get_map(self.current_map, [self.start_point], self.goal)]
        passage_points = list()
        for (cell_number, cell) in enumerate(self.trajectory[:-1]):
            passage_points.extend(get_intermediary_passage_points(cell, self.trajectory[cell_number + 1], self.current_map))
            frames.append(get_map(self.current_map, passage_points, self.goal))

        return frames


    def get_score(self):
        return len(self.trajectory) + np.linalg.norm(np.array(self.goal) - np.array(self.current_position))
