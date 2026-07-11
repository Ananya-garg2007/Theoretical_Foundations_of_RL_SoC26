import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Load Rewards
# ==========================================

rewards = np.load("Week_9/results/reward_history.npy")

# ==========================================
# Moving Average
# ==========================================

window = 20

moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    moving_average,
    linewidth=2,
    label="Moving Average Reward"
)

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.title("Deep Q-Network Training on CartPole")

plt.grid(True)

plt.legend()

plt.savefig("Week_9/results/dqn_learning_curve.png")

plt.show()