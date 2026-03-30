import numpy as np

# Random state
RANDOM_SEED = 42
RANDOM_STATE = np.random.default_rng(seed=RANDOM_SEED)

# Movement scale
STEP_SIZE = 10

# MATH
TWO_PI = 2 * np.pi

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

# Sampling
EPSILON = 1e-6
RELEVANCE_RADIUS = 100

# NRPA, GNRPA, ABGNRPA
LEARNING_RATE = 0.1
GAMMA = 1e-4
TAU = 10
HALF_LIFE_DIVIDER = 40
N_SAMPLES_TO_CHOOSE_FROM = 10

# Gaussian convolution
STATE_DISTANCE_PARAMETER = 50
ACTION_DISTANCE_PARAMETER = 50

# MCTS, RAVE and GRAVE
EXPLORATION_CONSTANT = np.sqrt(2)
N_DISCRETE_ACTIONS = 20
DISCRETE_ACTIONS = [
    round(action, 3) for action in np.arange(0, 1, 1 / N_DISCRETE_ACTIONS)
]
DISCRETE_MOVEMENTS = [
    [np.cos(angle * TWO_PI), np.sin(angle * TWO_PI)] for angle in DISCRETE_ACTIONS
]

# cMCTS and cRAVE/ cGRAVE parameters
PROGRESSIVE_WIDENING_PARAMETER = 0.05
BIAS_VALUE = 1e-6
N_VISITS_REFERENCE = 50

# cNMCTS
BANDWIDTH = 40
