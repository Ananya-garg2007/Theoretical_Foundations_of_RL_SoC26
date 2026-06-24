# Week 2: Reinforcement Learning Algorithms and Markov Decision Processes

---

## 1. Markov Decision Processes (MDPs)

A Markov Decision Process (MDP) provides a mathematical framework for sequential decision-making under uncertainty. It is defined as a tuple:

[
(S, A, P, R, \gamma)
]

* ( S ): state space
* ( A ): action space
* ( P(s' | s, a) ): transition probability
* ( R(s, a) ): reward function
* ( \gamma \in [0,1] ): discount factor

---

### Intuition

At every step:

* The agent is in a **state**
* Takes an **action**
* Receives a **reward**
* Moves to a **new state**

This interaction continues over time.

---

### Markov Property

$$
P(S_{t+1} \mid S_t, A_t, \dots, S_0, A_0) = P(S_{t+1} \mid S_t, A_t)
$$

**Meaning:**
The current state contains all relevant information about the past. The future depends only on the present state.

---

### Policy

A policy ( \pi ) defines the agent’s behaviour:

* Deterministic: ( a = \pi(s) )
* Stochastic: ( \pi(a|s) = P(A = a \mid S = s) )

👉 A policy is simply the agent’s **decision-making rule**.

---

## 2. Return and Objective

The total reward accumulated over time is called the **return**:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

---

### Intuition

* Immediate rewards matter more
* Future rewards are discounted

---

### Objective

$$
\pi^{*} = \arg\max_\pi \mathbb{E}_\pi [G_t]
$$

**Goal:**
Find a policy that maximizes expected total reward.

---

## 3. Value Functions and Bellman Equations

To evaluate how good states and actions are, we define value functions.

---

### State-Value Function

$$
V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
$$

**Meaning:**
Expected total reward starting from state ( s ).

---

### Action-Value Function

$$
Q^\pi(s,a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
$$

**Meaning:**
Expected reward if we take action ( a ) in state ( s ).

---

### Key Insight

* ( V(s) ): how good a state is
* ( Q(s,a) ): how good an action is

👉 Q-functions directly help in **decision-making**.

---

## Bellman Expectation Equation (State Value)

$$
V^\pi(s) = \mathbb{E}*\pi \big[ R*{t+1} + \gamma V^\pi(S_{t+1}) \big]
$$

Expanded form:

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\big[ R(s,a) + \gamma V^\pi(s') \big]
$$

---

### Intuition

> Value today = reward now + value of next state

---

## Bellman Expectation Equation (Action Value)

$$
Q^\pi(s,a) = \mathbb{E} \big[ R_{t+1} + \gamma \mathbb{E}*{a' \sim \pi}[Q^\pi(S*{t+1}, a')] \big]
$$

---

## 4. Derivation of Bellman Equation

Starting from:

$$
V^\pi(s) = \mathbb{E}[G_t]
$$

Expand return:

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

Substitute:

$$
V^\pi(s) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1})]
$$

---

## 5. Monte Carlo Methods

Monte Carlo methods estimate value functions using **complete episodes**.

---

### Return

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

---

### Update Rule

$$
V(s) \leftarrow V(s) + \alpha (G_t - V(s))
$$

---

### Intuition

> Play the full episode → then update

---

### Properties

* Model-free
* Unbiased
* High variance

---

## 6. Temporal Difference Learning and Q-Learning

Temporal Difference (TD) learning updates values using current estimates instead of waiting for full episodes.

---

### TD Update

$$
V(s) \leftarrow V(s) + \alpha [R + \gamma V(s') - V(s)]
$$

---

### Intuition

> Learn step-by-step using partial information

---

## Q-Learning

Q-learning is a TD control algorithm that learns the optimal action-value function.

---

### Bellman Optimality Equation

$$
Q^{*}(s, a) = \mathbb{E}[ R_{t+1} + \gamma \max_{a'} Q^{*}(s', a') ]
$$

---

### Update Rule

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]
$$

---

### Intuition

* Try an action
* Observe reward
* Improve estimate

---

### Key Concepts

* Bootstrapping
* Off-policy learning
* Convergence to optimal policy

---

## Exploration vs Exploitation

To balance learning:

* With probability (1 - \epsilon): exploit
* With probability ( \epsilon ): explore

---

## 7. From Q-Values to Policy

$$
\pi^{*}(s) = \arg\max_a Q^{*}(s,a)
$$

---

### Meaning

Choose the action with the highest expected reward.

---

### Example

| Action | (Q^*(s,a)) |
| ------ | ---------- |
| Left   | 4          |
| Right  | 12         |
| Back   | 1          |

Optimal action = **Right**

---

## 8. Deep Q-Networks (DQN)

Q-tables do not scale to large problems.

---

### Idea

Use neural networks:

$$
Q(s,a;\theta)
$$

---

### Loss Function

$$
L(\theta) = \mathbb{E}[(r + \gamma \max Q(s',a';\theta^-) - Q(s,a;\theta))^2]
$$

---

### Stabilization Techniques

* Experience Replay
* Target Networks

---

## 9. Policy Gradient and PPO

Instead of learning values, policy gradient methods directly optimize the policy.

---

### Objective

$$
J(\theta) = \mathbb{E}*{\pi*\theta}[G_t]
$$

---

### Gradient

$$
\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) Q^\pi(s,a)]
$$

---

## PPO (Proximal Policy Optimization)

### Objective

$$
L^{PPO}(\theta) = \mathbb{E}\left[\min\left(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A\right)\right]
$$

---

### Intuition

> Prevent the policy from changing too much in one update

---

## 10. DDPG

Used for continuous action spaces.

---

### Structure

* Actor: ( \mu(s) )
* Critic: ( Q(s,a) )

---

### Idea

Actor chooses actions, critic evaluates them.

---

## 11. Soft Actor-Critic (SAC)

SAC adds entropy to encourage exploration.

---

### Objective

$$
J(\pi) = \mathbb{E} [R(s,a) + \alpha \mathcal{H}(\pi)]
$$

---

### Entropy

$$
\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a|s)]
$$

---

### Intuition

> Encourage randomness to avoid premature convergence

---

## Conclusion

Reinforcement learning combines:

* Mathematical modeling (MDPs)
* Recursive reasoning (Bellman equations)
* Learning algorithms (Monte Carlo, TD, Q-learning)
* Deep learning (DQN, PPO, SAC)

---

### Key Insight

Most RL algorithms repeatedly:

* Estimate values
* Improve decisions
* Balance exploration and stability

---
