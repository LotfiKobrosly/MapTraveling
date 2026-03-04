import os
import warnings
import random
import numpy as np
import pandas as pd

from utils.sampling_utils import *
from utils.constants import *
from utils.map_random_generator import *
from utils.map_utils import *
from classes.path_generator import PathGenerator

warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    strategies = ["random_walk", "nrpa", "gnrpa", "abgnrpa", "mcts", "crave", "cgrave"]
    strategies = ["random_walk", "mcts", "crave", "cgrave"]
    n_obstacles_max = 5
    height_bounds = [100, 300]
    width_bounds = [100, 300]
    trajectory_max_length = 800
    n_maps = 1
    n_runs = 1
    inputs = {"level": 1}

    # Figures saving main directory
    figures_directory = "./figures"
    if not os.path.exists(figures_directory):
        os.mkdir(figures_directory)

    # Saving scores
    mean_score = np.zeros((n_maps, len(strategies)))
    std_score = np.zeros((n_maps, len(strategies)))
    min_score = np.zeros((n_maps, len(strategies)))

    for map_id in range(n_maps):
        print("\nRun n°", str(map_id + 1), ":...")

        # Figures for map
        map_figures_directory = figures_directory + "/" + "map_" + str(map_id)
        if not os.path.exists(map_figures_directory):
            os.mkdir(map_figures_directory)

        # Map dimensions
        height = 100
        width = 100
        current_map = generate_random_map_with_rectangular_obstacles(
            height, width, n_obstacles_max
        )
        start_point, goal = generate_start_and_end_points(current_map)

        for strategy_id, strategy in enumerate(strategies):
            # Strategy per map saving directory
            strategy_map_directory = map_figures_directory + "/" + strategy.upper()
            scores_list = list()
            if not os.path.exists(strategy_map_directory):
                os.mkdir(strategy_map_directory)

            # Running solver
            for run in range(n_runs):
                path_generator = PathGenerator(
                    current_map, start_point, goal, trajectory_max_length, strategy
                )
                if strategy == "random_walk":
                    inputs["n_iterations"] = 1000
                elif strategy in ["mcts", "crave", "cgrave"]:
                    inputs["n_iterations"] = 200000
                else:
                    inputs["n_iterations"] = 100
                path_generator.run(inputs)
                score = path_generator.get_score()
                scores_list.append(score)
                play_scenario(path_generator.get_movement_frames(), strategy_map_directory + "/run_" + str(run), score)
            mean_score[map_id, strategy_id] = np.mean(scores_list)
            std_score[map_id, strategy_id] = np.std(scores_list)
            min_score[map_id, strategy_id] = np.min(scores_list)
            print(strategy.upper(), " runs done")

    writer = pd.ExcelWriter("Aggregated_scores.xlsx", engine="xlsxwriter")

    mean_dataframe = pd.DataFrame(data=mean_score, columns=strategies, index=range(1, n_maps + 1))
    std_dataframe = pd.DataFrame(data=std_score, columns=strategies, index=range(1, n_maps + 1))
    min_dataframe = pd.DataFrame(data=min_score, columns=strategies, index=range(1, n_maps + 1))

    mean_dataframe.to_excel(writer, sheet_name="Mean score")
    std_dataframe.to_excel(writer, sheet_name="Standard deviation of score")
    min_dataframe.to_excel(writer, sheet_name="Min score")

    writer.close()

