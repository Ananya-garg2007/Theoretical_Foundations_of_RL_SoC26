import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from network import DQN
from replay_buffer import ReplayBuffer

# ==========================================================
# Environment
# ==========================================================

env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ==========================================================
# Hyperparameters
# ==========================================================

episodes = 500

batch_size = 64

gamma = 0.99

learning_rate = 0.001

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

target_update = 20

# ==========================================================
# Networks
# ==========================================================

online_network = DQN(state_size, action_size)

target_network = DQN(state_size, action_size)

target_network.load_state_dict(
    online_network.state_dict()
)

# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.Adam(
    online_network.parameters(),
    lr=learning_rate
)

loss_function = nn.MSELoss()

# ==========================================================
# Replay Buffer
# ==========================================================

buffer = ReplayBuffer(10000)

# ==========================================================
# Store Rewards
# ==========================================================

episode_rewards = []

# ==========================================================
# Training
# ==========================================================

for episode in range(episodes):

    state, info = env.reset()

    done = False

    total_reward = 0

    while not done:

        # --------------------------------------
        # ε-Greedy Action Selection
        # --------------------------------------

        if random.random() < epsilon:

            action = env.action_space.sample()

        else:

            state_tensor = torch.FloatTensor(state)

            with torch.no_grad():

                q_values = online_network(state_tensor)

            action = torch.argmax(q_values).item()

        # --------------------------------------
        # Environment Step
        # --------------------------------------

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # --------------------------------------
        # Store Experience
        # --------------------------------------

        buffer.push(

            state,

            action,

            reward,

            next_state,

            done

        )

        # --------------------------------------
        # Learn
        # --------------------------------------

        if len(buffer) >= batch_size:

            batch = buffer.sample(batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))

            actions = torch.LongTensor(actions)

            rewards = torch.FloatTensor(rewards)

            next_states = torch.FloatTensor(np.array(next_states))

            dones = torch.FloatTensor(dones)

            # -----------------------------
            # Current Q-values
            # -----------------------------

            current_q = online_network(states)

            current_q = current_q.gather(

                1,

                actions.unsqueeze(1)

            ).squeeze()

            # -----------------------------
            # Bellman Target
            # -----------------------------

            with torch.no_grad():

                next_q = target_network(next_states)

                max_next_q = next_q.max(1)[0]

                target_q = rewards + gamma * max_next_q * (1 - dones)

            # -----------------------------
            # Loss
            # -----------------------------

            loss = loss_function(

                current_q,

                target_q

            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        state = next_state

        total_reward += reward

    # ======================================================
    # End of Episode
    # ======================================================

    episode_rewards.append(total_reward)

    epsilon = max(

        epsilon_min,

        epsilon * epsilon_decay

    )

    if (episode + 1) % target_update == 0:

        target_network.load_state_dict(

            online_network.state_dict()

        )

    if (episode + 1) % 10 == 0:

        average_reward = np.mean(episode_rewards[-10:])

        print(

            f"Episode {episode+1:3d}"

            f" | Avg Reward: {average_reward:6.2f}"

            f" | Epsilon: {epsilon:.3f}"

        )

# ==========================================================
# Save Model
# ==========================================================

torch.save(

    online_network.state_dict(),

    "Week_9/results/dqn_model.pth"

)

np.save(

    "Week_9/results/reward_history.npy",

    np.array(episode_rewards)

)

print("\nTraining Complete!")

env.close()