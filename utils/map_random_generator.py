import random
import numpy as np

from utils.map_utils import draw_obstacles


def generate_random_map_with_rectangular_obstacles(height, width, n_obstacles):
    
    obstacles_corner_list = list()
    obstacles_height_list = list()
    obstacles_width_list = list()

    # Obstacles
    for i in range(n_obstacles):
        obstacles_corner_list.append([random.randint(0, height), random.randint(0, width)])
        obstacles_height_list.append(min(obstacles_corner_list[i][0] + random.randint(0, height // 5), height))
        obstacles_width_list.append(min(obstacles_corner_list[i][1] + random.randint(0, width // 5), width))
    # print(obstacles_corner_list[0])

    return draw_obstacles(
        np.zeros((height, width), dtype=np.int32),
        obstacles_corner_list,
        obstacles_height_list,
        obstacles_width_list,
    )


def generate_start_and_end_points(empty_map):

    height, width = np.shape(empty_map)
    # Start and goal, outside of obstacles
    start_point, goal = [0, 0], [0, 0]
    while np.linalg.norm(np.array(start_point) - (np.array(goal)) < 0.5 * min(height, width)):
        while True:
            start_point = [random.randint(0, height - 1), random.randint(0, width - 1)]
            if empty_map[start_point[0], start_point[1]] == 0:
                break
        while True:
            goal = [random.randint(0, height - 1), random.randint(0, width - 1)]
            if (empty_map[goal[0], goal[1]] == 0):
                break
    print("Start and goal generated")
    print("Start at: ", start_point)
    print("Goal at: ", goal)
    return start_point, goal
