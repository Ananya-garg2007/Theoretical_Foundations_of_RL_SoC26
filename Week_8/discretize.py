import gymnasium as gym
import numpy as np

# Create environment
env = gym.make("CartPole-v1")

# Number of bins for each state variable
NUM_BINS = (10, 10, 10, 10)

# Lower bounds
LOW = np.array([
    -4.8,
    -3.0,
    -0.418,
    -4.0
])

# Upper bounds
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


state, info = env.reset()

print("Original State:")
print(state)

print()

print("Discrete State:")
print(discretize_state(state))

env.close()