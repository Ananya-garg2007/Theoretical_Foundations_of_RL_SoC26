# Deep Q-Network (DQN)

After implementing Tabular Q-Learning, I observed that the agent was able to learn, but the performance quickly plateaued around a reward of 20.

Initially, I thought there might be an error in my implementation. However, after studying the theory, I realized that this was actually a limitation of **Tabular Q-Learning** itself.

---

# Why Does Q-Learning Fail?

Q-Learning stores all Q-values inside a **Q-table**.

```text
             Left     Right

State 1        2.1      5.4

State 2        8.3      4.7

State 3        1.6      7.2
```

This works well only when the number of states is reasonably small.

However, CartPole does **not** have discrete states.

A state consists of four continuous variables:

- Cart Position
- Cart Velocity
- Pole Angle
- Pole Angular Velocity

Mathematically,

$$
s=(x,\dot{x},\theta,\dot{\theta})
$$

Since each variable can take infinitely many values,

the number of possible states is also infinite.

Therefore,

it is impossible to store every state inside a Q-table.

---

# The Idea Behind DQN

Instead of storing every Q-value inside a table,

we train a **Neural Network** to approximate the Q-function.

Instead of

$$
Q(s,a)
$$

we now write

$$
Q(s,a;\theta)
$$

where

- $s$ = current state
- $a$ = action
- $\theta$ = parameters (weights) of the neural network

Instead of learning entries of a table,

we learn the weights of the neural network.

This allows the model to generalize to states that it has never seen before.

---

# Neural Network Architecture

The neural network receives the current state as input.

```text
Cart Position
Cart Velocity
Pole Angle
Angular Velocity
        │
        ▼
 Hidden Layer
        │
 Hidden Layer
        │
        ▼
Q(Left)   Q(Right)
```

Input Layer

- 4 neurons
- One for each state variable

Hidden Layers

- Learn useful representations of the environment

Output Layer

- 2 neurons
- One Q-value for each possible action

The agent simply chooses the action with the larger predicted Q-value.

---

# Bellman Equation

One of the biggest insights I gained during this project was that

> **The Bellman Equation does not change.**

The Bellman Target is still

$$ Target = R+\gamma\max_{a'}Q(s',a') $$

The only difference is that

- In Q-Learning, Q-values come from a Q-table.
- In DQN, Q-values are predicted by a neural network.

---

# Loss Function

Since the neural network predictions are initially inaccurate,

we calculate the error between

- Predicted Q-value
- Bellman Target

using the Mean Squared Error loss.

$$ Loss = (Target-Prediction)^2
$$

The optimizer then performs **Backpropagation** to update the neural network weights.

Gradually,

the predicted Q-values become closer to the Bellman Target.

---

# Experience Replay

Training directly from consecutive experiences causes instability because consecutive states are highly correlated.

Instead,

every interaction is stored inside a memory buffer.

```text
(State, Action, Reward, Next State)

↓

Replay Buffer

↓

Random Mini-batch

↓

Train Network
```

Random sampling breaks the correlation between experiences and improves learning stability.

---

# Target Network

Another problem arises if the same network computes both

- Prediction
- Target

because the target keeps changing while the network is trying to learn.

To solve this,

DQN maintains **two neural networks**.

### Online Network

- Updated after every gradient step

### Target Network

- Updated only periodically

The target network provides relatively stable Bellman targets,

making training much more stable.

---

# Complete DQN Pipeline

The overall training process is

```text
Observe Current State
        │
        ▼
Neural Network predicts Q-values
        │
        ▼
Choose Action (ε-greedy)
        │
        ▼
Environment returns
Reward + Next State
        │
        ▼
Store Experience
in Replay Buffer
        │
        ▼
Sample Random Mini-batch
        │
        ▼
Compute Bellman Target
        │
        ▼
Compute Loss
        │
        ▼
Backpropagation
        │
        ▼
Update Neural Network
        │
        ▼
Repeat
```

---

# Results

The Deep Q-Network significantly outperformed Tabular Q-Learning.

| Algorithm | Approximate Reward |
|-----------|-------------------:|
| Tabular Q-Learning | ~20 |
| Deep Q-Network | ~125 |

The improvement comes from replacing the Q-table with a neural network capable of learning directly from continuous state spaces.

---

# Key Takeaways

- Q-Learning stores Q-values inside a table.
- Q-tables cannot handle continuous state spaces efficiently.
- DQN replaces the Q-table with a neural network.
- The Bellman Equation remains exactly the same.
- Replay Buffers improve stability by randomizing training data.
- Target Networks stabilize Bellman targets during training.
- DQN learns much better policies for environments with continuous states.