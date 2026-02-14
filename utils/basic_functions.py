import numpy as np

def code_action(angle: float):
    return round(angle, 3)

def compute_heuristic_value(position: tuple, goal: tuple, angle: float) -> float:
    vector = np.array(list(goal)) - np.array(list(position))
    return (vector @ np.array([np.cos(angle), np.sin(angle)]).T) / np.linalg.norm(vector)
