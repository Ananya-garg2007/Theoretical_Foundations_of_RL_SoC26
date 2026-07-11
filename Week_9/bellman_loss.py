import torch
import torch.nn as nn
import torch.optim as optim

from network import DQN


# -----------------------------------
# Create Network
# -----------------------------------

model = DQN(state_size=4, action_size=2)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_function = nn.MSELoss()


# -----------------------------------
# Example Transition
# -----------------------------------

state = torch.tensor(
    [0.1, -0.2, 0.03, 0.15],
    dtype=torch.float32
)

next_state = torch.tensor(
    [0.12, -0.15, 0.05, 0.20],
    dtype=torch.float32
)

action = 0

reward = 1.0

gamma = 0.99


# -----------------------------------
# Forward Pass
# -----------------------------------

current_q_values = model(state)

predicted_q = current_q_values[action]


with torch.no_grad():

    next_q_values = model(next_state)

    max_next_q = torch.max(next_q_values)

    target = reward + gamma * max_next_q


# -----------------------------------
# Compute Loss
# -----------------------------------

loss = loss_function(predicted_q, target)

print("Predicted Q :", predicted_q.item())

print("Target Q    :", target.item())

print("Loss        :", loss.item())


# -----------------------------------
# Gradient Descent
# -----------------------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nOne gradient descent step completed.")