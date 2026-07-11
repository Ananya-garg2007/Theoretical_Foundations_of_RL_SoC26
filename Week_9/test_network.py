import torch

from network import DQN


state_size = 4
action_size = 2

model = DQN(state_size, action_size)


state = torch.tensor(
    [0.1, -0.2, 0.03, 0.15],
    dtype=torch.float32
)

output = model(state)

print("State:")
print(state)

print()

print("Predicted Q-values:")
print(output)

print()

print("Best Action:")
print(torch.argmax(output).item())