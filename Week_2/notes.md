# Week 2: Reinforcement Learning Algorithms and Markov Decision Processes

## 1. Markov Decision Processes (MDPs)

A Markov Decision Process (MDP) provides a mathematical framework for sequential decision-making under uncertainty. It is defined as a tuple:

(S, A, P, R, γ)

- S: state space  
- A: action space  
- P(s' | s, a): transition probability  
- R(s, a): reward function  
- γ ∈ [0,1]: discount factor  

The defining property of an MDP is the **Markov property**:

$$
P(S_{t+1} \mid S_t, A_t, \dots, S_0, A_0) = P(S_{t+1} \mid S_t, A_t)
$$

This means that the current state is a sufficient summary of the past.

A policy $\pi$ defines the agent’s behaviour:
- Deterministic: $a = \pi(s)$  
- Stochastic: $\pi(a|s) = P(A = a \mid S = s)$  

---

## 2. Return and Objective

The central objective in reinforcement learning is to maximize the expected cumulative reward, known as the return:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The discount factor γ ensures that future rewards are weighted less than immediate rewards and guarantees convergence of the sum for γ < 1.
The goal is to find an optimal policy π* such that:

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi [G_t]
$$

Thus, reinforcement learning can be viewed as an optimization problem over policies.

---

## 3. Value Functions and Bellman Equations

To evaluate policies, we define value functions.

The **state-value function**:

$$
V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
$$

and represents the expected return starting from state s under policy π. Basically, expected total reward starting from state s

The **action-value function**:

Expected reward if you take action a in state s 

$$
Q^\pi(s,a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
$$


These functions satisfy recursive relationships known as the Bellman expectation equations.

### Bellman Expectation Equation (State Value)

For the state-value function:

$$
V^\pi(s) = \mathbb{E}_\pi \big[ R_{t+1} + \gamma V^\pi(S_{t+1}) \big]
$$

Expanding the expectation explicitly:

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\big[ R(s,a) + \gamma V^\pi(s') \big]
$$

### Bellman Expectation Equation (Action Value)

$$
Q^\pi(s,a) = \mathbb{E} \big[ R_{t+1} + \gamma \mathbb{E}_{a' \sim \pi}[Q^\pi(S_{t+1}, a')] \big]
$$

These equations form the backbone of almost all RL algorithms, as they express value functions recursively in terms of themselves.

---

## 4. Monte Carlo Methods

Monte Carlo methods estimate value functions using complete episodes.

For a given episode, the return is computed as:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

The value estimate is updated using:

$$
V(s) \leftarrow V(s) + \alpha (G_t - V(s))
$$

where α is the learning rate.

Monte Carlo methods are:

- Model-free (no need for transition probabilities)
- Unbiased (true returns used)
- High variance (due to randomness in full episodes)
- They are suitable for episodic tasks but cannot be applied directly to continuing tasks without modification. 

---

## 5. Temporal Difference Learning and Q-Learning

Temporal Difference (TD) learning combines ideas from Monte Carlo methods and dynamic programming.

Q-learning is a fundamental TD control algorithm that learns the optimal action-value function directly.

It is based on the Bellman optimality equation:


$$
Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(s',a')]
$$

Update rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]
$$

This update uses bootstrapping, as it updates estimates using other learned estimates.

Key ideas:
- Bootstrapping  
- Off-policy learning(learns optimal policy independent of behavior) 
- Proven to converge under suitable conditions

---

## 6. Deep Q-Networks (DQN)

In large or continuous state spaces, storing Q-values in a table is infeasible. Deep Q-Networks approximate Q(s, a) using a neural network:

$$
Q(s,a;\theta)
$$

The parameters θ are learned by minimizing the loss:

Loss function:

$$
L(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2\right]
$$

To stabilize training, DQN uses:

- Experience replay (random sampling of past transitions)
- Target networks (fixed parameters θ⁻ for stable targets)


---

## 7. Policy Gradient and PPO

Instead of learning value functions, policy gradient methods directly optimize the policy.

The objective is:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]
$$

Using the likelihood ratio trick, the gradient becomes:

$$
\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) Q^\pi(s,a)]
$$

Proximal Policy Optimization (PPO) improves stability by limiting policy updates:

$$
L^{PPO}(\theta) = \mathbb{E}\left[\min\left(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A\right)\right]
$$

where:

$$
r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
$$

This clipping prevents excessively large updates, ensuring stable learning.

---

## 8. Deep Deterministic Policy Gradient (DDPG)

DDPG is an actor-critic algorithm designed for continuous action spaces.

- Actor: $\mu(s)$  outputs actions
- Critic: $Q(s,a)$  evaluates actions

The policy gradient is:

$$
\nabla_\theta J \approx \mathbb{E}[\nabla_a Q(s,a)\big|_{a=\mu(s)} \nabla_\theta \mu(s)]
$$

DDPG combines ideas from deterministic policy gradients and Q-learning.
---

## 9. Soft Actor-Critic (SAC)

SAC extends actor-critic methods by incorporating entropy maximization.

The objective is:

$$
J(\pi) = \mathbb{E} \left[\sum (R(s,a) + \alpha \mathcal{H}(\pi(\cdot|s)))\right]
$$

where entropy is:

$$
\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}[\log \pi(a|s)]
$$

Maximizing entropy encourages exploration and prevents premature convergence to suboptimal policies.

---

## Conclusion

Reinforcement learning can be formalized using MDPs and solved using value-based and policy-based methods. The Bellman equations provide the theoretical backbone, while algorithms such as Q-learning, PPO, DDPG, and SAC offer practical solutions across different problem settings.

A key insight is that most RL methods rely on recursive estimation of value functions and iterative improvement of policies, balancing exploration, stability, and convergence.