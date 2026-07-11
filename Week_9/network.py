"""
Neural Network

Input Layer
4 neurons

↓

Hidden Layer
128 neurons
(ReLU)

↓

Hidden Layer
128 neurons
(ReLU)

↓

Output Layer
2 neurons

(Q(left), Q(right))
"""

import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),     #ReLU adds non-linearity, ReLU(x)=max(0,x). Without it, the network would just behave like one giant linear equation.

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)

        )

    def forward(self, x):
        return self.network(x)