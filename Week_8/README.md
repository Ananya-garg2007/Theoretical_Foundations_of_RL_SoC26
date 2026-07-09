# Week 8: Tabular Q-Learning on CartPole

## Objective

The objective of this week's project was to bridge the gap between the mathematical theory of Reinforcement Learning and its practical implementation.

Instead of using existing RL libraries, the goal was to implement the **Tabular Q-Learning algorithm from scratch**, understand every component mathematically, and observe how an agent gradually learns optimal behaviour through interaction with the environment.

---

# Environment

- **Environment:** CartPole-v1
- **Library:** Gymnasium
- **State Space:** Continuous (4-dimensional)
- **Action Space:** Discrete

Actions:

- **0:** Push cart to the left
- **1:** Push cart to the right

The agent receives a reward of **+1** for every timestep the pole remains balanced.

---

# State Representation

Since Tabular Q-Learning requires discrete states, the continuous observation space was discretized.

The four state variables are:

1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Angular Velocity

Each variable was divided into **10 bins**, producing a discrete state representation of the form

```
(position_bin,
 velocity_bin,
 angle_bin,
 angular_velocity_bin)
```

For example,

```
Continuous State

[-0.02, 0.31, 0.05, -0.16]

↓

Discrete State

(4, 6, 5, 3)
```

This discrete state serves as the index into the Q-table.

---

# Q-Table

A Q-table stores the expected long-term reward of taking every possible action from every discrete state.

```
Q[state][action]
```

Shape of the Q-table:

```
(10 × 10 × 10 × 10 × 2)
```

where

- 10 bins for each of the four state variables
- 2 possible actions

Initially,

```
Q(s,a)=0
```

for every state-action pair.

---

# Exploration Strategy

An **ε-greedy policy** was used.

With probability ε:

- choose a random action (exploration)

Otherwise:

- choose the action with the highest Q-value (exploitation)

The exploration probability decays throughout training:

```
ε ← max(0.01, ε × 0.995)
```

This allows the agent to explore early and exploit learned knowledge later.

---

# Bellman Update

The Q-table is updated using the Bellman Optimality Equation:

\[
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
r
+
\gamma
\max_{a'}Q(s',a')
-
Q(s,a)
\right]
\]

where

- α : learning rate
- γ : discount factor
- r : immediate reward

This update gradually moves the Q-value toward the observed target.

---

# Training Procedure

For every episode:

1. Reset the environment
2. Observe the current state
3. Discretize the state
4. Select an action using ε-greedy
5. Execute the action
6. Observe the next state and reward
7. Update the Q-table using the Bellman equation
8. Repeat until the episode terminates

The total reward obtained during each episode is recorded.

---

# Results

Training was performed for **5000 episodes**.

The learning curve shows a gradual increase in average reward during the early stages of training before stabilizing.

The following artifacts are generated after training:

- `q_table.npy`
- `reward_history.npy`
- `q_learning_curve.png`

---

# Discussion

Although the implementation correctly follows the Tabular Q-Learning algorithm, performance eventually plateaus around an average reward of approximately **20**.

This occurs because CartPole has a **continuous state space**, while Tabular Q-Learning requires discrete states. Discretization inevitably loses information, causing many distinct physical states to share the same discrete representation.

As a result, the learned policy cannot achieve optimal performance.

This limitation motivates **Deep Q-Networks (DQN)**, where a neural network is used to approximate the action-value function instead of maintaining a discrete lookup table.

---

# Repository Structure

```
Week_8/
│
├── train_qlearning.py
├── discretize.py
├── plots.py
├── notes.md
├── project_plan.md
├── README.md
│
├── Results/
│   ├── q_table.npy
│   ├── reward_history.npy
│   └── q_learning_curve.png
│
├── network.py
├── replay_buffer.py
└── utils.py
```

---

# Key Concepts Learned

- Markov Decision Process (MDP)
- State and Action spaces
- Return
- Q-Function
- Bellman Optimality Equation
- Temporal Difference Learning
- ε-greedy Exploration
- Exploration vs Exploitation
- Tabular Reinforcement Learning
- Limitations of Tabular Methods

---

# Future Work

The next stage of the project will replace the tabular Q-function with a neural network implementation (Deep Q-Network), allowing reinforcement learning to scale to high-dimensional continuous state spaces while preserving the Bellman update framework.