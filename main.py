import os
import time
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
    # strategies = ["random_walk", "nrpa", "gnrpa", "abgnrpa", "mcts", "crave", "cgrave"]
    # strategies = ["random_walk", "mcts", "crave", "cgrave"]
    strategies = ["random_walk", "nrpa", "gnrpa", "abgnrpa"]
    n_obstacles_max = 5
    height_bounds = [100, 300]
    width_bounds = [100, 300]
    trajectory_max_length = 400
    n_maps = 10
    n_runs = 10
    inputs = {"level": 1}

    # Figures saving main directory
    figures_directory = "./figures"
    if not os.path.exists(figures_directory):
        os.mkdir(figures_directory)

    # Saving scores
    mean_score = np.zeros((n_maps, len(strategies)))
    std_score = np.zeros((n_maps, len(strategies)))
    min_score = np.zeros((n_maps, len(strategies)))
    average_time = np.zeros((n_maps, len(strategies)))

    for map_id in range(n_maps):
        print("\nMap n°", str(map_id + 1))

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
            time_list = list()
            if not os.path.exists(strategy_map_directory):
                os.mkdir(strategy_map_directory)

            print("Running", strategy.upper())
            # Running solver
            for run in range(n_runs):
                try:
                    print("Run n°: ", run + 1)
                    start_time = time.time()
                    path_generator = PathGenerator(
                        current_map, start_point, goal, trajectory_max_length, strategy
                    )
                    if strategy == "random_walk":
                        inputs["n_iterations"] = 10000
                    elif strategy in ["mcts", "crave", "cgrave"]:
                        inputs["n_iterations"] = 20000
                    else:
                        inputs["n_iterations"] = 200
                    path_generator.run(inputs)
                    score = path_generator.best_score
                    scores_list.append(score)
                    time_list.append(time.time() - start_time)
                    play_scenario(
                        [path_generator.get_trajectory_frame()],
                        strategy_map_directory + "/run_" + str(run),
                        score,
                    )
                except KeyboardInterrupt:
                    continue
            mean_score[map_id, strategy_id] = np.mean(scores_list)
            std_score[map_id, strategy_id] = np.std(scores_list)
            min_score[map_id, strategy_id] = np.min(scores_list)
            average_time[map_id, strategy_id] = np.mean(time_list)
            print(strategy.upper(), " runs done")

    writer = pd.ExcelWriter("Aggregated_scores.xlsx", engine="xlsxwriter")

    mean_dataframe = pd.DataFrame(
        data=mean_score, columns=strategies, index=range(1, n_maps + 1)
    )
    std_dataframe = pd.DataFrame(
        data=std_score, columns=strategies, index=range(1, n_maps + 1)
    )
    min_dataframe = pd.DataFrame(
        data=min_score, columns=strategies, index=range(1, n_maps + 1)
    )
    time_dataframe = pd.DataFrame(
        data=average_time, columns=strategies, index=range(1, n_maps + 1)
    )

    mean_dataframe.to_excel(writer, sheet_name="Mean score")
    std_dataframe.to_excel(writer, sheet_name="Standard deviation of score")
    min_dataframe.to_excel(writer, sheet_name="Min score")
    time_dataframe.to_excel(writer, sheet_name="Average time")

    writer.close()
