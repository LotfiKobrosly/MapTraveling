from classes import *
from solvers.algorithms import *

def run_solver(path_generator: PathGenerator, inputs: dict):
    height, width = path_generator.current_map.shape
    if path_generator.strategy == "random_walk":
        trajectories, scores = list(), list()
        for _ in range(inputs["n_iterations"]):
            path_generator.reinitialize()
            generate_path(path_generator)
            trajectories.append(path_generator.trajectory[:])
            scores.append(path_generator.get_score())
        best_score_index = np.argmin(scores)
        path_generator.trajectory = trajectories[best_score_index]
        path_generator.best_score = scores[best_score_index]
    elif path_generator.strategy in ["nrpa", "gnrpa", "abgnrpa"]:

        max_radius = np.sqrt(height**2 + width**2)
        relevance_levels = [0.5]
        radius = 0.5
        while radius < max_radius:
            radius *= 2
            relevance_levels.append(radius)
        if path_generator.strategy == "abgnrpa":
            heuristic_values = HeuristicValues(bias_factor=inputs.get("bias_factor", 1))
        else:
            heuristic_values = None
        nrpa(
            path_generator,
            level=inputs["level"],
            n_policies=inputs["n_policies"],
            policy=dict(),
            relevance_radius_list=relevance_levels,
            heuristic_values=heuristic_values,
        )
    elif path_generator.strategy == "pbrnrpa":
        policy = {
            (0, height, 0, width): {
                "move": RANDOM_STATE.uniform(-1, 1, size=2),
                "n_visits": 0,
                "threshold": 1,
            }
        }
        nrpa(path_generator, level=inputs["level"], n_policies=inputs["n_policies"], policy=policy)
    elif path_generator.strategy in ["mcts", "rave", "grave"]:
        mcts(path_generator, n_iterations=inputs["n_iterations"])
    elif path_generator.strategy in ["cmcts", "crave", "cgrave"]:
        cmcts(path_generator, n_iterations=inputs["n_iterations"])
    elif path_generator.strategy == "cnmcts":
        trajectory, actions, score = cnmcts(
            path_generator,
            level=inputs["level"],
            bandwidth=inputs["bandwidth"],
        )
        path_generator.trajectory = trajectory
        path_generator.actions = actions
        path_generator.best_score = score
    elif path_generator.strategy == "crbnmcts":
        trajectory, actions, score = crbnmcts(
            path_generator,
            level=inputs["level"],
            bandwidth=inputs["bandwidth"],
        )
        path_generator.trajectory = trajectory
        path_generator.actions = actions
        path_generator.best_score = score
    else:
        raise ValueError("Strategy ill-defined: " + path_generator.strategy)