"""
Implements the continuous Policy By Region NRPA
"""

import numpy as np

from scipy.ndimage import gaussian_filter

from utils.basic_functions import code
from utils.sampling_utils import *
from utils.map_utils import cell_is_reachable, continuous_cell_selector, get_region_area, cell_is_within_rectangle
from utils.constants import (
    RELEVANCE_RADIUS,
    RANDOM_STATE,
    LEARNING_RATE,
)

def pbrnrpa_step(position: tuple, current_map: np.ndarray, policy: dict, sampling_radius: float=0.001):
    relevant_regions = [
        key
        for key in policy.keys()
        if cell_is_within_rectangle(position, key)
    ]
    chosen_region = relevant_regions[
        np.argmin([get_region_area(bounding_coordinates) for bounding_coordinates in relevant_regions])
    ]
    mean = policy[chosen_region]["move"]
    policy[chosen_region]["n_visits"] += 1
    new_cell = [-1, -1]
    widening = 0.01  # to prevent the search from being stuck
    while not cell_is_reachable(position, new_cell, current_map):
        move = RANDOM_STATE.normal(mean, sampling_radius + widening, size=2)
        widening += 0.01
        if int(widening / 0.01) % 100 == 0:
            print("Mean ", mean)
            print("Sigma: ", sampling_radius + widening_factor)
            print("Reached ", int(widening / 0.01), " iterations of widening")
        new_cell = continuous_cell_selector(position, move)
    return move, new_cell


def adapt_pbrnrpa_policy(
    best_trajectory: list,
    best_course_of_actions: list,
    policy: dict,
    learning_rate: float
):
    # Going through the trajectory, affecting them to their corresponding regions
    regional_division_of_points = {
        key: list()
        for key in policy.keys()
    }
    for point_index, point in enumerate(best_trajectory[:-1]):
        for key in regional_division_of_points.keys():
            if cell_is_within_rectangle(point, key):
                regional_division_of_points[key].append(best_course_of_actions[point_index])

    for region in regional_division_of_points.keys():
        regional_division_of_points[region] = np.mean(list(regional_division_of_points[region]), axis=0)

    # Adapting the assigned moves to each region
    for region in regional_division_of_points.keys():
        policy[region]["move"] += learning_rate * (regional_division_of_points[region] - policy[region]["move"])
    
    return policy
    
