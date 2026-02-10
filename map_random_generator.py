import random
import numpy as np


def generate_random_map_with_rectangular_obstacles(height, width, n_obstacles):
    # Initialize empty map
    empty_map = np.zeros((height, width), dtype=np.int32)

    # Obstacles
    for _ in range(n_obstacles):
        obstacle_corner = [random.randint(0, height), random.randint(0, width)]
        obstacle_height = random.randint(0, height // 5)
        obstacle_width = random.randint(0, width // 5)
        for abcissae in range(
            obstacle_corner[0], min(obstacle_corner[0] + obstacle_height, height)
        ):
            for ordinate in range(
                obstacle_corner[1], min(obstacle_corner[1] + obstacle_width, width)
            ):
                empty_map[abcissae, ordinate] = 1

    return empty_map


def generate_start_and_end_points(empty_map):

    height, width = np.shape(empty_map)
    # Start and goal, outside of obstacles
    while True:
        start_point = [random.randint(0, height - 1), random.randint(0, width - 1)]
        if empty_map[start_point[0], start_point[1]] == 0:
            break
    while True:
        goal = [random.randint(0, height - 1), random.randint(0, width - 1)]
        if empty_map[goal[0], goal[1]] == 0:
            break
    return start_point, goal
