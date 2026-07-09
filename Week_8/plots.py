import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("Week_8/Results/reward_history.npy")

window = 100

moving_average = np.convolve(
    rewards,
    np.ones(window)/window,
    mode="valid"
)

plt.figure(figsize=(10,5))

plt.plot(moving_average)

plt.xlabel("Episode")

plt.ylabel("Average Reward")

plt.title("Q-Learning on CartPole")

plt.grid(True)

plt.savefig("Week_8/Results/q_learning_curve.png")

plt.show()