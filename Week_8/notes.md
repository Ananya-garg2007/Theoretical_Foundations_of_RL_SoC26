# Week 8

Goal

Implement two reinforcement learning algorithms

1. Tabular Q-learning
2. Deep Q-Network

Environment:
CartPole-v1

Objective:
Understand how Bellman equations translate into actual code.

**Pipeline**

Environment

↓

Current State

↓

Agent

↓

Action

↓

Environment

↓

Reward + Next State

↓

Bellman Update

↓

Repeat

# Week 8: CartPole Environment Exploration

## Goal

Before implementing any reinforcement learning algorithm, it is important to understand the environment in which the agent operates. In this project, the chosen environment is **CartPole-v1**, a classic control problem available in Gymnasium.

The objective of this week is to understand the state space, action space, reward structure, and the complete interaction loop between the agent and the environment.

---

# CartPole-v1 Environment

CartPole consists of a cart that can move horizontally on a track with a pole attached to it by a frictionless joint.

Initially, the pole starts close to the upright position. The agent must apply forces to the cart so that the pole remains balanced for as long as possible.

The environment follows the standard Reinforcement Learning framework:

- Agent
- Environment
- State
- Action
- Reward

---

# State Space

The environment returns a state consisting of four continuous values.

State =

1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Angular Velocity

Example:

[-0.0004, -0.0325, 0.0099, 0.0388]

Each observation completely describes the current situation of the cart and pole.

Since these values are continuous, there are infinitely many possible states.

---

# Observation Space

The observation space is represented as

Box(low, high, (4,), float32)

which means:

- Continuous state space
- Four-dimensional observation vector

Approximate limits are

Cart Position:
[-4.8, 4.8]

Pole Angle:
[-0.418 rad, 0.418 rad]

The velocity components are theoretically unbounded.

---

# Action Space

The action space is

Discrete(2)

There are only two possible actions.

Action 0:
Push the cart to the left.

Action 1:
Push the cart to the right.

Thus, at every timestep, the agent chooses one of these two actions.

---

# Reward Function

The environment gives

Reward = +1

for every timestep that the pole remains balanced.

Therefore, maximizing the total reward is equivalent to keeping the pole upright for as long as possible.

---

# Episode Termination

An episode terminates if any one of the following occurs:

- The pole falls beyond the allowed angle.
- The cart moves outside the allowed position.
- The maximum episode length is reached.

After termination, the environment is reset and a new episode begins.

---

# Agent–Environment Interaction

Every reinforcement learning algorithm follows the same interaction cycle.

Current State

↓

Agent selects an Action

↓

Environment executes the Action

↓

Environment returns

- Reward
- Next State
- Done flag

↓

Agent updates its policy or value function

↓

Repeat until the episode terminates.

---

# Why CartPole?

CartPole is widely used because:

- Simple state space
- Small action space
- Fast training
- Easy visualization
- Suitable for testing new RL algorithms

It provides an excellent starting point for implementing both classical and deep reinforcement learning algorithms.

---

# Why Tabular Q-Learning Cannot Be Used Directly

Q-learning stores one value for every (state, action) pair inside a Q-table.

However, CartPole has a continuous state space, meaning there are infinitely many possible states.

A Q-table cannot store infinitely many entries.

Therefore, before implementing tabular Q-learning, the continuous observations must be converted into a finite number of discrete states. This process is known as **state discretization**.

Later in this project, Deep Q-Networks (DQN) will overcome this limitation by replacing the Q-table with a neural network that approximates the action-value function.

---

# Learning Outcome

By exploring the CartPole environment, I understood:

- How an RL environment is represented.
- What information is contained in the state.
- How actions influence the environment.
- How rewards are generated.
- Why continuous state spaces require function approximation or state discretization.
- How the agent and environment interact during every episode.

## State Discretization

CartPole observations are continuous, whereas tabular Q-learning requires a finite number of discrete states.

To bridge this gap, each state variable is divided into a fixed number of bins. Before discretization, the velocity components are clipped to reasonable ranges because their theoretical limits are infinite.

The discretization process consists of three steps:

1. Clip each state variable to predefined limits.
2. Normalize the values between 0 and 1.
3. Convert the normalized values into integer bin indices.



For example,

Continuous State

[-0.01, -0.02, 0.03, 0.01]

↓

Discrete State

(4, 4, 5, 4)

The resulting tuple can now be used as an index in a Q-table.