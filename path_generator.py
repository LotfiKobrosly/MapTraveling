import random
from copy import deepcopy
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from utils import *

RANDOM_STATE = np.random.default_rng(seed=42)


def conditional_gaussian_1d(mu, Sigma, x_fixed):  # ChatGPT
    """
    mu: (3,)
    Sigma: (3, 3)
    x_fixed: (2,) -> [x1, x2]
    """
    mu_a = mu[:2]
    mu_b = mu[2]

    Sigma_aa = Sigma[:2, :2]
    Sigma_ab = Sigma[:2, 2]
    Sigma_ba = Sigma[2, :2]
    Sigma_bb = Sigma[2, 2]

    Sigma_aa_inv = np.linalg.inv(Sigma_aa)

    cond_mean = mu_b + Sigma_ba @ Sigma_aa_inv @ (x_fixed - mu_a)
    cond_var = Sigma_bb - Sigma_ba @ Sigma_aa_inv @ Sigma_ab

    return cond_mean, cond_var


def sample_conditional_gmm_sklearn(gmm, x_fixed, n_samples=1):  # ChatGPT
    """
    gmm: fitted sklearn GaussianMixture (3D)
    x_fixed: array-like shape (2,) -> fixed x1, x2
    """
    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_

    K = len(weights)

    # --- update mixture weights ---
    log_weights = np.zeros(K)
    for k in range(K):
        mu_a = means[k][:2]
        Sigma_aa = covs[k][:2, :2]

        log_weights[k] = np.log(weights[k]) + multivariate_normal.logpdf(
            x_fixed, mu_a, Sigma_aa
        )

    # normalize safely
    log_weights -= log_weights.max()
    new_weights = np.exp(log_weights)
    new_weights /= new_weights.sum()

    # --- sample component indices ---
    components = np.random.choice(K, size=n_samples, p=new_weights)

    # --- sample x3 ---
    samples = np.zeros(n_samples)
    for i, k in enumerate(components):
        mean_k, var_k = conditional_gaussian_1d(means[k], covs[k], x_fixed)
        samples[i] = np.random.normal(mean_k, np.sqrt(var_k))

    return samples


class PathGenerator(object):

    def __init__(self, current_map, start_point, goal, trajectory_size, strategy):
        self.current_map = current_map
        self.start_point = start_point
        self.goal = goal
        self.trajectory_size = trajectory_size
        self.strategy = strategy
        self.current_position = start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]
        self.actions = list()
        height, width = current_map.shape
        self.best_score = height * width
        if strategy == "nrpa":
            self.policy = dict()
            """[
                [
                    dict() for j in range(width)
                ]
                for i in range(height)
            ]"""
            self.nrpa_iterations = 0

    def is_finished(self):
        return (
            (self.goal[0] == self.current_position[0])
            and (self.goal[1] == self.current_position[1])
            or (self.current_steps == self.trajectory_size)
        )

    def reinitialize(self):
        self.current_position = self.start_point
        self.current_steps = 0
        self.trajectory = [self.start_point]

    def code_action(self, angle: float):
        return round(angle, 3)

    def step(self):
        if self.strategy == "random_walk":
            normalized_angle = random.random()

        elif self.strategy == "nrpa":
            position = self.current_position
            policy = self.policy
            if policy:
                mixture_data = np.concatenate(
                    [
                        RANDOM_STATE.multivariate_normal(
                            np.array(list(key)),
                            value,
                            size=max(10, 100 - self.nrpa_iterations),
                        )
                        for (key, value) in policy.items()
                    ]
                ).reshape(-1, 3)
                model = GaussianMixture(3).fit(mixture_data)
                normalized_angle = sample_conditional_gmm_sklearn(model, list(position))
            else:
                normalized_angle = random.random()

        else:
            raise (ValueError("No strategy defined"))

        return normalized_angle, cell_selector(
            self.current_position, normalized_angle * 2 * np.pi
        )

    def adapt_policy(
        self, best_trajectory, best_course_of_actions, policy=None, best_score=None
    ):
        if policy is None:
            policy = self.policy
        for point_index, point in enumerate(best_trajectory[:-1]):
            i, j = point[0], point[1]
            current_policy = policy
            if current_policy:
                average_covariance = np.mean(
                    np.array(list(current_policy.values())), axis=0
                )
                for key, value in current_policy.items():
                    current_policy[key] = value * 10 / 9
                if (
                    current_policy.get(
                        (i, j, self.code_action(best_course_of_actions[point_index])),
                        None,
                    )
                    is None
                ):
                    current_policy[
                        (i, j, self.code_action(best_course_of_actions[point_index]))
                    ] = (average_covariance * 9 / 10)
                else:
                    current_policy[
                        (i, j, self.code_action(best_course_of_actions[point_index]))
                    ] *= (9 / 10) ** 2
                normalizing_factor = np.sum(
                    np.array(list(current_policy.values())), axis=0
                )
                for key, value in current_policy.items():
                    current_policy[key] = value / normalizing_factor

            else:
                height, width = self.current_map.shape
                current_policy[
                    (i, j, self.code_action(best_course_of_actions[point_index]))
                ] = RANDOM_STATE.uniform([0, 0, 0], [1, 1, 1], size=(3, 3))

    def nrpa(self, level: int = 1, n_policies: int = 100):

        if level == 0:
            self.generate_path()
            self.nrpa_iterations += 1

        else:
            iteration_number = self.nrpa_iterations
            best_score = self.best_score
            best_trajectory = deepcopy(self.trajectory)
            best_course_of_actions = deepcopy(self.actions)
            policy = deepcopy(self.policy)
            for _ in range(n_policies):
                self.reinitialize()
                self.nrpa(level - 1, n_policies)
                score = self.get_score()
                if score < best_score:
                    best_score = score
                    best_trajectory = deepcopy(self.trajectory)
                    best_course_of_actions = deepcopy(self.actions)
                self.adapt_policy(best_trajectory, best_course_of_actions)
            self.nrpa_iterations = iteration_number + 1
            self.adapt_policy(
                best_trajectory,
                best_course_of_actions,
                policy=policy,
                best_score=best_score,
            )
            self.policy = policy
            self.trajectory = best_trajectory
            self.best_score = best_score

    def generate_path(self):
        height, width = self.current_map.shape

        self.trajectory = [self.start_point]
        while not self.is_finished():
            next_cell_found = False
            while not next_cell_found:
                angle, new_cell = self.step()
                if (
                    (new_cell[0] >= 0)
                    and (new_cell[0] < height)
                    and (new_cell[1] >= 0)
                    and (new_cell[1] < width)
                    and (self.current_map[new_cell[0], new_cell[1]] != 1)
                ):
                    next_cell_found = True
                    intermediary_passage_points = get_intermediary_passage_points(
                        self.current_position, new_cell, self.current_map
                    )
                    for point in intermediary_passage_points:
                        if self.current_map[point[0], point[1]] == 1:
                            next_cell_found = False
                            break
                    if not next_cell_found:
                        continue
                    break

            self.current_steps += 1
            self.current_position = new_cell
            self.trajectory.append(new_cell)
            self.actions.append(angle)

    def get_movement_frames(self):
        frames = [get_map(self.current_map, [self.start_point], self.goal)]
        passage_points = list()
        for cell_number, cell in enumerate(self.trajectory[:-1]):
            passage_points.extend(
                get_intermediary_passage_points(
                    cell, self.trajectory[cell_number + 1], self.current_map
                )
            )
            frames.append(get_map(self.current_map, passage_points, self.goal))

        return frames

    def get_score(self):
        return len(self.trajectory) + np.linalg.norm(
            np.array(self.goal) - np.array(self.current_position)
        )
