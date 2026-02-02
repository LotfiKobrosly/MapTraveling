import numpy as np
import matplotlib.pyplot as plt

STEP_SIZE = 5


def get_map(current_map, passage_points, goal):
    height, width = np.shape(current_map)
    image_to_show = np.ones((height, width, 3), dtype=np.int32)
    last_point = passage_points[-1]
    finished = (last_point[0] == goal[0]) and (last_point[1] == goal[1])
    for i in range(height):
        for j in range(width):
            if current_map[i, j] == 1:
                image_to_show[i, j] = np.array([0, 0, 0])
            if (i == goal[0]) and (j == goal[1]):
                image_to_show[i, j] = np.array([125, 255, 125])
            for point in passage_points:
                if (i == point[0]) and (j == point[1]):
                    if finished:
                        image_to_show[i, j] = np.array([125, 255, 125])
                    else:
                        image_to_show[i, j] = np.array([255, 0, 0])
            if (
                (image_to_show[i, j, 0] == 1)
                and (image_to_show[i, j, 1] == 1)
                and (image_to_show[i, j, 2] == 1)
            ):
                image_to_show[i, j] = [255, 255, 255]

    return image_to_show


def cell_selector(position, angle):
    return (
        int(np.cos(angle) * STEP_SIZE) + position[0],
        int(np.sin(angle) * STEP_SIZE) + position[1],
    )

def get_intermediary_passage_points(start: tuple, end:tuple, current_map: np.ndarray):
    intermediary_passage_points = list()
    smaller_start_x, bigger_start_x = start[0], end[0]
    smaller_start_y, bigger_start_y = start[1], end[1]
    if smaller_start_x > bigger_start_x:
        smaller_start_x, bigger_start_x = (
            bigger_start_x,
            smaller_start_x,
        )
    if smaller_start_y > bigger_start_y:
        smaller_start_y, bigger_start_y = (
            bigger_start_y,
            smaller_start_y,
        )
    for i in range(smaller_start_x, bigger_start_x + 1):
        for j in range(smaller_start_y, bigger_start_y + 1):
            intermediary_passage_points.append([i, j])
    return intermediary_passage_points



def play_scenario(frames):
    fig = plt.figure()
    viewer = fig.add_subplot(111)
    fig.show()  # Initially shows the figure
    # loop over your images
    for frame in frames:
        viewer.clear()  # Clears the previous image
        viewer.imshow(frame)  # Loads the new image
        plt.pause(0.1)  # Delay in seconds
        fig.canvas.draw()  # Draws the image to the screen
    fig.savefig("./last_path.jpeg")
