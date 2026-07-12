# Understanding Q-Learning Before the Implementation

Before writing any code, I wanted to understand **what Reinforcement Learning is**, **how Q-Learning works mathematically**, and **why the algorithm is able to learn an optimal policy**. This document summarizes the concepts I learned before implementing Q-Learning from scratch.

---

# What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of Machine Learning where an **agent learns by interacting with an environment**.

Unlike supervised learning, where the correct answers are already available, the agent learns through **trial and error**. Every action receives feedback in the form of a reward, and over time the agent discovers which actions maximize the total reward.

The interaction follows a continuous cycle:

```text
Current State
      ↓
Choose an Action
      ↓
Environment Responds
      ↓
Receive Reward
      ↓
Move to Next State
      ↓
Repeat
```

The objective is **not simply to maximize the immediate reward**, but rather to maximize the **total future reward**.

---

# The CartPole Problem

For this project, I used the **CartPole-v1** environment provided by Gymnasium.

The environment consists of a cart that can move either left or right while balancing a pole.

At every time step,

- The agent observes the current state.
- The agent chooses either **Left** or **Right**.
- The environment returns the next state.
- The agent receives a reward of **+1** for every time step the pole remains balanced.

The objective is therefore very simple:

> **Keep the pole balanced for as long as possible.**

---

# What is a State?

A **state** completely describes the current situation of the environment.

For CartPole, a state consists of four continuous variables:

- Cart Position
- Cart Velocity
- Pole Angle
- Pole Angular Velocity

Mathematically,

$$
s=(x,\dot{x},\theta,\dot{\theta})
$$

Every decision made by the agent depends entirely on the current state.

---

# What is an Action?

For CartPole there are only two possible actions.

```text
0 → Move Left

1 → Move Right
```

The objective of the agent is to determine **which action is better in every possible state.**

---

# What is a Q-value?

Before implementing Q-Learning, the first question I asked was:

> **How do we know whether moving left or right is a good decision?**

This is exactly what a **Q-value** represents.

Mathematically,

$$
Q(s,a)
$$

where

- $s$ represents the current state
- $a$ represents an action

A Q-value estimates the **expected long-term reward** obtained by taking action $a$ in state $s$ and then following the optimal policy afterwards.

Simply put,

> **A Q-value tells us how good an action is in a particular state.**

---

# Example

Suppose the pole is leaning slightly to the left.

The Q-values are

| State | Left | Right |
|------|------:|------:|
| Pole slightly left | 3 | 12 |

This means

- Moving Left has an expected future reward of **3**
- Moving Right has an expected future reward of **12**

Since

$$
12 > 3
$$

the optimal action is

```text
Move Right
```

because it is expected to keep the pole balanced for a longer period.

---

# What is a Q-table?

Since Q-Learning stores the Q-values explicitly, it maintains a **Q-table**.

- Rows represent **states**
- Columns represent **actions**

Initially,

```text
             Left     Right

State 1        0         0

State 2        0         0

State 3        0         0

State 4        0         0
```

Every value starts at **zero** because the agent has never interacted with the environment.

The Q-table therefore acts as the **memory of the agent**.

As training progresses, the table gradually becomes more accurate.

---

# How Does the Q-table Learn?

Suppose the agent is currently in

```text
State 5
```

The agent chooses

```text
Move Right
```

The environment returns

```text
Reward = +1
```

and the agent reaches

```text
State 6
```

Now suppose the largest Q-value in State 6 is

```text
8
```

Notice something important.

The agent **does not simply store the immediate reward**.

Instead it asks

> **How valuable is the next state?**

This leads to the Bellman Target.

---

# Bellman Target

The Bellman Target is

$$
Target = R+\gamma\max_aQ(s',a)
$$

where

- $R$ is the immediate reward
- $\gamma$ is the discount factor
- $Q(s',a)$ is the value of the best action in the next state

Suppose

```text
Reward = 1

Best Future Q-value = 8

γ = 0.9
```

Then

$$
Target
=
1+0.9(8)
=
8.2
$$

The Bellman Target estimates

> **What should the current Q-value ideally become?**

It combines

- Immediate reward
- Future reward

into a single target.

---

# Updating the Q-table

Suppose

```text
Old Q-value = 5

Target = 8.2

Learning Rate = 0.1
```


The Q-value is updated using the Q-learning update rule:

```math
Q_{\text{new}}
=
Q_{\text{old}}
+
\alpha
\left(
\text{Target}
-
Q_{\text{old}}
\right)

=> 
Q_{\text{new}}
=
5+0.1(8.2-5)
=
5.32

```

Notice that we **do not replace** the old value.

Instead,

the algorithm moves a **small step** towards the Bellman Target.

Repeating this update thousands of times allows the Q-values to converge.

---

# How Does the Q-table Become Intelligent?

Initially,

```text
             Left     Right

State 5        0         0
```

After many interactions,

```text
             Left     Right

State 5       2.4       9.1
```

The table now contains useful information.

Whenever the agent visits State 5,

it simply selects the action with the highest Q-value.

Mathematically,

## \(\pi^*(s)\)

The optimal policy chooses the action with the highest Q-value:

$$
\pi^*(s)
=
\arg\max_a Q(s,a)
$$
This means

> **Among all possible actions in the current state, choose the action with the highest Q-value.**

At this point,

the Q-table itself represents the learned policy.

---

# Limitation of Tabular Q-Learning

At this stage I realized an important limitation.

A Q-table only works when the number of states is reasonably small.

However,

CartPole has **continuous state variables**.

This means there are infinitely many possible states.

Since storing infinitely many rows inside a table is impossible,

I had to **discretize** the state space.

Discretization groups many different continuous states into the same discrete state, causing information loss.

As a result,

Tabular Q-Learning is unable to achieve very high performance on CartPole.

This naturally motivated the transition to **Deep Q-Networks**, where the Q-table is replaced by a neural network capable of approximating Q-values directly from continuous states.

---

# Key Takeaways

- Reinforcement Learning learns through interaction with an environment.
- Q-values estimate the long-term usefulness of an action.
- The Q-table stores these Q-values for every state-action pair.
- The Bellman equation updates Q-values using both immediate and future rewards.
- Repeated Bellman updates gradually populate the Q-table.
- The learned policy simply chooses the action with the highest Q-value.
- Tabular Q-Learning struggles with continuous state spaces because a Q-table cannot represent infinitely many states.
- This limitation motivates the use of **Deep Q-Networks (DQN)**.
