import numpy as np

# Random state
RANDOM_SEED = 42
RANDOM_STATE = np.random.default_rng(seed=RANDOM_SEED)

# Movement scale
STEP_SIZE = 2

# Algorithms
ALGORITHMS = [
    "random_walk",
    "mcts",
    "rave",
    "grave",
    "cmcts",
    "crave",
    "cgrave",
    "nrpa",
    "gnrpa",
    "abgnrpa",
]

# Gaussian Mixture
N_GMM_COMPONENTS = 10
EPSILON = 1e-6
RELEVANCE_RADIUS = 10

# NRPA, GNRPA, ABGNRPA
LEARNING_RATE = 0.1
GAMMA = 0.01
TAU = 10
N_SAMPLES_TO_CHOOSE_FROM = 20
SAMPLING_METHODS = [
    "GaussianMixture",
    "LinearRegression",
    "RidgeRegrression",
    "MLP",
    "XGBoost",
]
MLP_SHAPE = (100,)
MLP_LOSS = "squared_error"
MLP_ACTIVATION = "relu"
MLP_SOLVER = "adam"
MLP_LEARNING_RATE = 0.001
MLP_PARAMETERS = {
    "loss": MLP_LOSS,
    "learning_rate": "constant",
    "learning_rate_init": MLP_LEARNING_RATE,
    "hidden_layer_sizes": MLP_SHAPE,
    "activation": MLP_ACTIVATION,
    "solver": MLP_SOLVER,
}

# Gaussian convolution
STATE_DISTANCE_PARAMETER = 50
ACTION_DISTANCE_PARAMETER = 50

# MCTS, RAVE and GRAVE
EXPLORATION_CONSTANT = np.sqrt(2)
N_DISCRETE_ACTIONS = 20
DISCRETE_ACTIONS = [
    round(action, 3) for action in np.arange(0, 1, 1 / N_DISCRETE_ACTIONS)
]

# cMCTS and cRAVE/ cGRAVE parameters
PROGRESSIVE_WIDENING_PARAMETER = 0.05
BIAS_VALUE = 1e-6
N_VISITS_REFERENCE = 50
