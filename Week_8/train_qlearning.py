import gymnasium as gym
import numpy as np
import random

from discretize import discretize_state

# ======================================================
# Create Environment
# ======================================================

env = gym.make("CartPole-v1")

# ======================================================
# Hyperparameters
# ======================================================

NUM_BINS = (10, 10, 10, 10)
NUM_ACTIONS = env.action_space.n

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 0.2        # Initial exploration probability
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 5000

# ======================================================
# Initialize Q-table
# ======================================================

Q = np.zeros(NUM_BINS + (NUM_ACTIONS,))

# Store reward obtained in every episode
episode_rewards = []

# ======================================================
# Training Loop
# ======================================================

for episode in range(episodes):

    # Reset environment
    state, info = env.reset()

    # Convert continuous state to discrete state
    state = discretize_state(state)

    done = False

    episode_reward = 0

    while not done:

        # ==========================================
        # ε-Greedy Action Selection
        # ==========================================

        if random.random() < epsilon:
            action = env.action_space.sample()      # Explore
        else:
            action = np.argmax(Q[state])            # Exploit

        # ==========================================
        # Take Action
        # ==========================================

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = discretize_state(next_state)

        # ==========================================
        # Bellman Update
        # ==========================================

        best_future = np.max(Q[next_state])

        target = reward + gamma * best_future

        td_error = target - Q[state][action]

        Q[state][action] += alpha * td_error

        # ==========================================
        # Move to Next State
        # ==========================================

        state = next_state

        episode_reward += reward

    # ==============================================
    # End of Episode
    # ==============================================

    episode_rewards.append(episode_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode+1:4d} | "
            f"Reward = {episode_reward:4.0f} | "
            f"Epsilon = {epsilon:.3f}"
        )

# ======================================================
# Save Results
# ======================================================

np.save("Week_8/Results/q_table.npy", Q)
np.save("Week_8/Results/reward_history.npy", np.array(episode_rewards))

print("\nTraining Complete!")

print(f"Average Reward: {np.mean(episode_rewards):.2f}")

env.close()