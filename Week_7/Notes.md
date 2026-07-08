# Week 7: Planning, Exploration, and Advanced Reinforcement Learning

## Introduction

In the previous weeks, we studied model-free reinforcement learning algorithms such as Q-learning, DQN, and REINFORCE, where the agent learns solely through interaction with the environment. While these methods have achieved remarkable success, they often require an enormous number of interactions before learning an effective policy.

This motivates several important questions:

* Can the agent use its previous experiences more efficiently?
* How should it balance exploration and exploitation?
* How can reinforcement learning scale to continuous and high-dimensional environments?

This week addresses these questions through planning methods, exploration strategies, and modern actor-critic algorithms.

---

## Part 1: Planning & Exploration

### 1. Model-Based Reinforcement Learning

In model-free reinforcement learning, the agent directly learns a policy or value function without explicitly understanding how the environment behaves. Model-based reinforcement learning takes a different approach. Instead of only learning the optimal policy, the agent also learns (or is given) a model of the environment consisting of:

* **Transition dynamics:** $P(s'|s,a)$
* **Reward function:** $R(s,a)$

Once this model is available, the agent can simulate future interactions internally without actually interacting with the real environment. This process is known as **planning**.

**Advantages include:**

* Better sample efficiency
* Faster learning
* Ability to reason about hypothetical situations

However, learning an accurate model can itself be difficult for complex environments.

### 2. Planning vs Learning

Learning and planning solve the same problem using different sources of information. Modern reinforcement learning often combines both.

**Learning**

* Learns directly from real interactions.
* Every update requires interaction with the environment.
* Usually more robust when the model is unknown.

**Planning**

* Learns using a model.
* Can generate imaginary experiences.
* Makes much better use of previous data.

### 3. Dyna-Q

One of the earliest algorithms combining planning and learning is Dyna-Q. The algorithm consists of three stages:

* **Step 1:** Interact with the real environment. Observe $(s,a,r,s')$ and update the Q-value using standard Q-learning.
* **Step 2:** Store this transition inside a learned model. The model gradually approximates $P(s'|s,a)$ and $R(s,a)$.
* **Step 3:** Randomly sample previously observed state-action pairs. Use the learned model to generate imaginary experiences. Update the Q-values again.

Thus, one real interaction produces many learning updates.

**Why Dyna-Q Works:**
Instead of waiting for new experiences, the agent repeatedly reuses previous knowledge. This significantly improves sample efficiency because every interaction contributes multiple updates.

> **The Dyna-Q Workflow:**
> Real Experience $\rightarrow$ Q-learning Update $\rightarrow$ Model Learning $\rightarrow$ Planning Updates $\rightarrow$ Improved Policy

### 4. Exploration vs Exploitation

One of the central challenges in reinforcement learning is balancing:

* **Exploration:** Trying actions whose outcomes are uncertain.
* **Exploitation:** Choosing the action currently believed to be the best.

Too much exploration wastes time, while too much exploitation may prevent discovering better policies.

### 5. $\epsilon$-Greedy Exploration

The simplest exploration strategy is $\epsilon$-greedy. The agent chooses:

* The best action with probability $1-\epsilon$
* A random action with probability $\epsilon$

**Advantages:** Extremely simple and computationally cheap.
**Limitations:** Random exploration ignores previous knowledge. Even obviously poor actions may still be selected.

### 6. Upper Confidence Bound (UCB)

UCB uses the principle of *optimism under uncertainty*. Instead of selecting actions solely based on expected reward, it also considers uncertainty. The action chosen is:

$$a = \arg\max_a \left( Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right)$$

where:

* $Q(a)$ is the estimated reward
* $N(a)$ is the number of times action $a$ has been selected
* $t$ is the current timestep
* $c$ controls exploration

Actions that have been explored less receive larger confidence bonuses.

**$\epsilon$-Greedy vs UCB**

| Feature | $\epsilon$-Greedy | UCB |
| --- | --- | --- |
| **Exploration Style** | Random exploration | Directed exploration |
| **Uncertainty** | Ignores uncertainty | Explicitly models uncertainty |
| **Complexity** | Very simple | More computationally expensive |
| **Sample Efficiency** | May waste samples | More sample efficient |

In practice, UCB generally achieves lower regret because it explores more intelligently.

### 7. Regret Minimization

The performance of an exploration strategy is often measured using regret. Regret is the difference between the reward obtained by the optimal policy and the reward actually collected by the learning algorithm. Mathematically:

$$Regret(T) = T\mu^* - \sum_{t=1}^{T}\mu_t$$

A good exploration algorithm minimizes cumulative regret over time.

---

## Part 2: Modern Deep RL Algorithms

### 8. Challenges in Deep Reinforcement Learning

Classical reinforcement learning algorithms assume relatively small state spaces, tabular value functions, and extensive exploration. Deep reinforcement learning introduces additional challenges:

* Extremely high-dimensional observations
* Sparse rewards
* Expensive exploration
* Long training times
* Instability during optimization

These limitations motivated the development of more sophisticated algorithms.

### 9. Actor-Critic Methods

Actor-Critic methods combine the advantages of value-based and policy-based learning. The architecture consists of two components:

* **Actor:** Learns the policy $\pi_\theta(a|s)$
* **Critic:** Estimates $V(s)$ or $Q(s,a)$

The critic evaluates actions while the actor improves the policy.

### 10. Proximal Policy Optimization (PPO)

PPO is one of the most widely used reinforcement learning algorithms today. Instead of allowing unrestricted policy updates, PPO limits how much the policy changes during each optimization step.

* **Advantages:** Stable training, easy implementation, strong empirical performance, and excellent sample efficiency.
* **Weaknesses:** Slower than deterministic methods and requires on-policy data.

### 11. Deep Deterministic Policy Gradient (DDPG)

DDPG extends deterministic policy gradients to continuous control.

* **Advantages:** Handles continuous action spaces, sample efficient, and off-policy.
* **Weaknesses:** Sensitive to hyperparameters, can become unstable, and exploration is often difficult.

### 12. Soft Actor-Critic (SAC)

SAC introduces entropy maximization into actor-critic learning. Instead of maximizing only reward, SAC maximizes **Reward + Entropy**. This encourages continual exploration.

* **Advantages:** Excellent exploration, highly stable, state-of-the-art performance, and sample efficient.
* **Weaknesses:** More computationally expensive and more complex to implement.

**Comparison of PPO, DDPG, and SAC**

| Feature | PPO | DDPG | SAC |
| --- | --- | --- | --- |
| **Learning Strategy** | On-policy | Off-policy | Off-policy |
| **Action Space** | Continuous & Discrete | Continuous | Continuous |
| **Stability** | Very High | Moderate | High |
| **Exploration** | Moderate | Weak | Excellent |
| **Sample Efficiency** | Moderate | High | Very High |
| **Complexity** | Low | Medium | High |

### Connecting Classical RL to Deep RL

Modern reinforcement learning can be viewed as a gradual evolution:

> Q-learning introduced value-based learning. $\rightarrow$ DQN replaced Q-tables with neural networks. $\rightarrow$ Policy Gradient methods optimized policies directly. $\rightarrow$ Actor-Critic combined both ideas. $\rightarrow$ PPO, DDPG, and SAC refined Actor-Critic methods to achieve greater stability, sample efficiency, and scalability.

Thus, modern deep reinforcement learning is not a completely different field—it is a natural extension of the classical reinforcement learning algorithms studied in earlier weeks.

## Conclusion

Week 7 brings together many of the central ideas in reinforcement learning. Model-based methods such as Dyna-Q demonstrate how planning can dramatically improve sample efficiency, while exploration strategies like $\epsilon$-greedy and UCB illustrate different approaches to balancing exploration and exploitation. Finally, actor-critic algorithms such as PPO, DDPG, and SAC show how reinforcement learning has evolved to solve complex, high-dimensional problems that were beyond the reach of classical tabular methods.