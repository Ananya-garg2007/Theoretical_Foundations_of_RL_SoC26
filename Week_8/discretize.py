import numpy as np

NUM_BINS = (10, 10, 10, 10)

LOW = np.array([
    -4.8,
    -3.0,
    -0.418,
    -4.0
])

HIGH = np.array([
    4.8,
    3.0,
    0.418,
    4.0
])


def discretize_state(state):
    """
    Convert a continuous CartPole state
    into a tuple of discrete indices.
    """

    state = np.clip(state, LOW, HIGH)

    ratios = (state - LOW) / (HIGH - LOW)

    discrete = (ratios * (np.array(NUM_BINS) - 1)).astype(int)

    return tuple(int(x) for x in discrete)