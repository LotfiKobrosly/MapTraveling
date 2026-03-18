import numpy as np


def code(array: tuple):
    return (round(array[0], 3), round(array[1], 3))


def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    return (vector_1 @ vector_2.T) / (
        np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    )


def compute_heuristic_value(position: tuple, goal: tuple, move: tuple) -> float:
    vector = np.array(list(goal)) - np.array(list(position))
    return cosine_similarity(vector, move)


def gaussian_kernel(sigma, radius=None):  # ChatGPT
    if radius is None:
        radius = int(3 * sigma)

    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel
