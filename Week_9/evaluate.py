import torch
import gymnasium as gym

from network import DQN

# ======================================
# Environment
# ======================================

env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ======================================
# Load Network
# ======================================

model = DQN(state_size, action_size)

model.load_state_dict(
    torch.load("Week_9/results/dqn_model.pth")
)

model.eval()

# ======================================
# Run Agent
# ======================================

state, info = env.reset()

done = False

total_reward = 0

while not done:

    state_tensor = torch.FloatTensor(state)

    with torch.no_grad():

        q_values = model(state_tensor)

    action = torch.argmax(q_values).item()

    state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    total_reward += reward

print()

print("Total Reward:", total_reward)

env.close()