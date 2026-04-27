import random
from copy import deepcopy
import numpy as np
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
        self.best_trajectory = [self.start_point]
        self.actions = list()
        self.best_course_of_actions = list()
        height, width = current_map.shape
        self.best_score = height * width
        self.score_normalizer = height * width
        if strategy in ["nrpa", "gnrpa", "abgnrpa", "pbrnrpa"]:
            self.area = height * width
            self.nrpa_iterations = 0
        elif strategy in ["mcts", "rave", "grave"]:
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
                        tuple(move): {
                            "n_visits": 0,
                            "cumulative_score": 0,
                        }
                        for move in DISCRETE_ACTIONS
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
        elif strategy in ["cnmcts", "crbnmcts"]:
            self.states_actions = dict()
        else:
            raise ValueError("Strategy ill-defined: " + str(strategy))

    def is_finished(self):
        return (
            np.linalg.norm(
                np.array(list(self.current_position)) - np.array(list(self.goal))
            )
            < 0.5
        ) or (self.current_steps >= self.trajectory_size)

    def reinitialize(self):
        self.current_position = self.start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]
        self.actions = list()
        if self.strategy in ["mcts", "rave", "grave", "crave", "cgrave"]:
            self.reference_state = self.start_point

    def get_score(self):
        return len(self.trajectory) * STEP_SIZE + np.linalg.norm(
            np.array(self.goal) - np.array(self.current_position)
        )

    def update(self, move, new_cell):
        self.current_position = code(new_cell)
        self.trajectory.append(code(new_cell))
        self.actions.append(code(move))
        self.current_steps += 1

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
