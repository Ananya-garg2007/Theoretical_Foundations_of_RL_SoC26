import torch

from network import DQN


state_size = 4
action_size = 2

# Online network
online = DQN(state_size, action_size)

# Target network
target = DQN(state_size, action_size)

# Copy weights
target.load_state_dict(online.state_dict())

print("Target network initialized!")

# ------------------------------------------------
# Change online network weights
# ------------------------------------------------

with torch.no_grad():

    for param in online.parameters():

        param.add_(1.0)

print("Online network changed.")

# ------------------------------------------------
# Synchronize target network
# ------------------------------------------------

target.load_state_dict(online.state_dict())

print("Target network synchronized.")