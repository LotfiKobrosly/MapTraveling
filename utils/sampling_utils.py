import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression, LogisticRegression, ridge_regression
from sklearn.neighbors import KNeighborsRegressor, KernelDensity
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor

from utils.constants import LEARNING_RATE, EPSILON, MLP_PARAMETERS

class GaussianKernel2D: #ChatGPT
    def __init__(self, center, sigma):
        """
        center: array-like of shape (2,) -> mean of the Gaussian
        sigma: float -> standard deviation (same for both dimensions)
        """
        self.mu = np.asarray(center, dtype=float)
        self.sigma = float(sigma)
        self.dim = 2
        
        # Precompute constants
        self.norm_const = 1.0 / ((2 * np.pi * self.sigma**2) ** (self.dim / 2))

    def pdf(self, x):
        """
        Compute probability density at point x.
        x: array-like of shape (2,)
        """
        x = np.asarray(x, dtype=float)
        diff = x - self.mu
        exponent = -0.5 * np.dot(diff, diff) / (self.sigma**2)
        return self.norm_const * np.exp(exponent)

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


def gradient_log_covariance(sample, center, covariance):  # ChatGPT
    size = sample.shape[0]
    try:
        inverted_covariance = np.linalg.inv(covariance)
    except:
        inverted_covariance = np.linalg.inv(covariance + EPSILON * np.eye(3))

    delta = (sample - center).reshape(size, 1)
    if np.isnan(delta).any():
        raise ValueError("Error is at delta: " + str(delta))

    term = inverted_covariance @ delta @ delta.T @ inverted_covariance
    if np.isnan(term).any():
        raise ValueError("Error is at term: " + str(delta))

    return 0.5 * (term - inverted_covariance)


def update_covariance(
    covariance, center, sample, score, learning_rate=LEARNING_RATE, epsilon=EPSILON
):  # ChatGPT
    nan_before = False
    if np.isnan(covariance).any():
        nan_before = True
    covariance_change = learning_rate * score * gradient_log_covariance(sample, center, covariance)
    covariance += covariance_change

    # Symmetrize
    covariance = (covariance + covariance.T) / 2

    # Numerical stability
    covariance += epsilon * np.eye(3)

    nan_after = False
    if np.isnan(covariance).any():
        nan_after = True

    if not nan_before and nan_after:
        print("NaN appears here")
        print("Learning rate: ", learning_rate)
        print("Score change: ", score)
        print("Center: ", center)
        print("Sample: ", sample)
        print("Gradient: ", gradient_log_covariance(sample, center, covariance))
    return covariance


def get_model(sampling_method: str = "LinearRegression"):
    match sampling_method:
        case "RidgeRegression":
            return ridge_regression()
        case "MLP":
            return MLPRegressor(**MLP_PARAMETERS)
        case "XGBoost":
            return XGBRegressor()
        case "KernelRidge":
            return KernelRidge()
        case "KNeighborsRegressor":
            return KNeighborsRegressor(weights="distance")
        case _:
            return LinearRegression()

