import warnings
import random
import numpy as np

from utils.sampling_utils import *
from utils.constants import *
from utils.map_random_generator import *
from utils.map_utils import *
from classes.path_generator import PathGenerator

warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    strategies = ["random_walk", "nrpa", "gnrpa", "abgnrpa"]
    n_obstacles_max = 5
    height_bounds = [100, 300]
    width_bounds = [100, 300]
    trajectory_max_length = 400

    current_map = generate_random_map_with_rectangular_obstacles(
        100, 100, n_obstacles_max
    )
    start_point, goal = generate_start_and_end_points(current_map)
    inputs = {
        "n_iterations": 100,
        "level": 1,
    }
    for strategy in strategies:
        path_generator = PathGenerator(
            current_map, start_point, goal, trajectory_max_length, strategy
        )
        path_generator.run(inputs)
        score = path_generator.get_score()
        play_scenario(path_generator.get_movement_frames(), strategy.upper(), score)
        print(strategy, " run done")

