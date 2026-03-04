import numpy as np


def code_action(angle: float):
    return round(angle, 3)


def code_state(position: list):
    return (code_action(position[0]), code_action(position[1]))


def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    return (vector_1 @ vector_2.T) / (
        np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    )


def compute_heuristic_value(position: tuple, goal: tuple, angle: float) -> float:
    return cosine_similarity(
        np.array(list(goal)) - np.array(list(position)),
        np.array([np.cos(angle), np.sin(angle)]),
    )


def gaussian_kernel(sigma, radius=None):  # ChatGPT
    if radius is None:
        radius = int(3 * sigma)

    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

