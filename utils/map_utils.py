import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.constants import STEP_SIZE
from utils.basic_functions import code


def get_map(current_map: np.ndarray, passage_points: list, goal: tuple) -> np.ndarray:
    height, width = np.shape(current_map)
    image_to_show = np.ones((height, width, 3), dtype=np.int32)
    last_point = passage_points[-1]
    finished = (np.linalg.norm(np.array(last_point) - np.array(goal)) < 0.5)
    for i in range(height):
        for j in range(width):
            if current_map[i, j] == 1:
                image_to_show[i, j] = np.array([0, 0, 0])
            if (i == goal[0]) and (j == goal[1]):
                for step_1 in range(-1, 2):
                    if (i + step_1 >= 0) and (i + step_1 < height):
                        for step_2 in range(-1, 2):
                            if (j + step_2 >= 0) and (j + step_2 < width):
                                image_to_show[i + step_1, j + step_2] = np.array(
                                    [125, 255, 125]
                                )

            for point in passage_points[1:]:
                if (i == point[0]) and (j == point[1]):
                    if finished:
                        image_to_show[i, j] = np.array([125, 255, 125])
                    else:
                        image_to_show[i, j] = np.array([255, 0, 0])
            start_point = passage_points[0]
            for step_1 in range(-1, 2):
                if (start_point[0] + step_1 >= 0) and (
                    start_point[0] + step_1 < height
                ):
                    for step_2 in range(-1, 2):
                        if (start_point[1] + step_2 >= 0) and (
                            start_point[1] + step_2 < width
                        ):
                            image_to_show[
                                start_point[0] + step_1, start_point[1] + step_2
                            ] = np.array([125, 125, 255])
            if (
                (image_to_show[i, j, 0] == 1)
                and (image_to_show[i, j, 1] == 1)
                and (image_to_show[i, j, 2] == 1)
            ):
                image_to_show[i, j] = [255, 255, 255]

    return image_to_show

def draw_obstacles(empty_map : np.ndarray, obstacles_corner_list : list, obstacles_height_list : list, obstacles_width_list : list):
    n_obstacles = len(obstacles_corner_list)
    assert (len(obstacles_width_list) == n_obstacles) and (len(obstacles_height_list) == n_obstacles), "Obstacles' lists' lenghths mistmatch"
    for obstacle in range(n_obstacles):
        obstacle_corner = obstacles_corner_list[obstacle]
        obstacle_height = obstacles_height_list[obstacle]
        obstacle_width = obstacles_width_list[obstacle]
        for abcissae in range(
            obstacle_corner[0], obstacle_height
        ):
            for ordinate in range(
                obstacle_corner[1], obstacle_width
            ):
                empty_map[abcissae, ordinate] = 1
    return empty_map

def load_map_with_rectangular_obstacles(filename: str):
    with open(filename, "r") as file:
        map_elements = json.load(file)

    height = map_elements["height"]
    width = map_elements["width"]
    start_point = map_elements["start_point"]
    goal = map_elements["goal"]
    n_obstacles = map_elements["n_obstacles"]
    obstacles_corner_list = map_elements["obstacles_corner_list"]
    obstacles_height_list = map_elements["obstacles_height_list"]
    obstacles_width_list = map_elements["obstacles_width_list"]

    return (
        draw_obstacles(
            np.zeros((height, width), dtype=np.int32),
            obstacles_corner_list,
            obstacles_height_list,
            obstacles_width_list,
        ),
        start_point,
        goal,
    )

def load_raw_map(json_filename: str, map_id: str):
    with open(json_filename, "r") as file:
        map_elements = json.load(file)[map_id]
    empty_map = 1 - np.array(Image.open(map_elements["map_file"]).convert('L')) / 255
    return (
        empty_map.round(),
        map_elements["start_point"],
        map_elements["goal"],
    )


def cell_selector(position: tuple, move: tuple)-> tuple:
    return (
        int(move[0] * STEP_SIZE) + position[0],
        int(move[1] * STEP_SIZE) + position[1],
    )

def continuous_cell_selector(position: tuple, movement: tuple) -> tuple:
    return code(np.array(position) + STEP_SIZE * np.array(movement))

def cell_is_within_rectangle(cell: tuple, bounding_coordinates: tuple) -> bool:
    x1, x2, y1, y2 = bounding_coordinates
    x, y = cell[0], cell[1]
    return ((x - x1) * (x - x2) <= 0) and ((y - y1) * (y - y2) <= 0)

def get_region_area(bounding_coordinates: tuple) -> float:
    x1, x2, y1, y2 = bounding_coordinates
    return np.absolute((x1 - x2) * (y1 - y2))

def subdivide_region(bounding_coordinates: tuple) -> list:
    x1, x2, y1, y2 = bounding_coordinates
    middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2
    return [
        (x1, middle_x, y1, middle_y),
        (x1, middle_x, middle_y, y2),
        (middle_x, x2, y1, middle_y),
        (middle_x, x2, middle_y, y2),
    ]

def get_intermediary_passage_points(start: tuple, end: tuple, current_map: np.ndarray) -> list:
    intermediary_passage_points = list()
    x0, x1 = int(start[0]), int(end[0])
    y0, y1 = int(start[1]), int(end[1])
    slope_numerator, slope_denominator = y1 - y0, x1 - x0
    if slope_numerator == 0 and slope_denominator == 0:
        intermediary_passage_points = [[x0, x1]]
    elif slope_numerator == 0:
        step = int((x1 - x0) / np.absolute(x1 - x0))
        if not step in [1, -1]:
            raise ValueError("Step value of intermediary passage points: " + str(step))
        for i in range(x0, x1 + step, step):
            intermediary_passage_points.append([i, y0])
    elif slope_denominator == 0:
        step = int((y1 - y0) / np.absolute(y1 - y0))
        if not step in [1, -1]:
            raise ValueError("Step value of intermediary passage points: " + str(step))
        for i in range(y0, y1 + step, step):
            intermediary_passage_points.append([x0, i])
    else:
        slope = slope_numerator / slope_denominator
        offset = y0 - x0 * slope
        step = int((x1 - x0) / np.absolute(x1 - x0))
        if not step in [1, -1]:
            raise ValueError("Step value of intermediary passage points: " + str(step))
        for i in range(x0, x1 + step, step):
            abcissae = int(slope * i + offset)
            if abcissae > max(y0, y1):
                abcissae = max(y0, y1)
            if abcissae < min(y0, y1):
                abcissae = min(y0, y1)
            intermediary_passage_points.append([i, abcissae])
    if len(intermediary_passage_points) >= 2:
        updated_passage_points = list()
        for point_index, point in enumerate(intermediary_passage_points[:-1]):
            updated_passage_points.append(point)
            if np.absolute(point[1] - intermediary_passage_points[point_index + 1][1]) > 1:
                start, end = point[1], intermediary_passage_points[point_index + 1][1]
                if start > end:
                    start, end = end, start
                for i in range(start, end):
                    updated_passage_points.append([point[0], i])
            updated_passage_points.append(intermediary_passage_points[point_index + 1])
        intermediary_passage_points = updated_passage_points[:]
    return intermediary_passage_points

def cell_is_reachable(current_position: tuple, new_cell: tuple, current_map: np.ndarray):
    height, width = current_map.shape
    if (
        (new_cell[0] >= 0)
        and (new_cell[0] < height)
        and (new_cell[1] >= 0)
        and (new_cell[1] < width)
        and (current_map[int(new_cell[0]), int(new_cell[1])] != 1)
    ):
        intermediary_passage_points = get_intermediary_passage_points(
            current_position, new_cell, current_map
        )
        for point in intermediary_passage_points:
            if current_map[point[0], point[1]] == 1:
                return False
    #    print("New cell found")
        return True
    return False

def play_scenario(frames: list, filename: str, score: float, wait_time: float=0.1):
    fig = plt.figure()
    viewer = fig.add_subplot(111)
    fig.show()  # Initially shows the figure
    # loop over your images
    for frame_number, frame in enumerate(frames):
        viewer.clear()  # Clears the previous image
        viewer.imshow(frame)  # Loads the new image
        plt.pause(wait_time)  # Delay in seconds
        fig.canvas.draw()  # Draws the image to the screen
    fig.suptitle("Score: " + "{:.3f}".format(score))
    fig.savefig(filename + ".jpeg")
    plt.close()
