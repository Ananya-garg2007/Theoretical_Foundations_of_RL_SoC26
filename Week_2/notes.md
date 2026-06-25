# Week 2: Reinforcement Learning Algorithms and Markov Decision Processes

**Project:** Theoretical Foundations of RL (SoC 2026)

**Topic:** 
- Q-learning and Monte Carlo methods
- Proximal Policy Optimization (PPO)
- Deep Deterministic Policy Gradient (DDPG)
- Soft Actor-Critic (SAC)
- Deep Q-Network (optional)
- MDPs: (S, A, P, R, gamma), Markov property, policies
- State-value and action-value functions

**Resources Covered:** 
- David Silver Lectures 1-2
- Sutton and Barto (2020), Ch. 2-3
- GeeksforGeeks (Q-learning, Monte Carlo): https://www.geeksforgeeks.org/q-learning-in-python/ 

---

## 1. Markov Decision Processes (MDPs)

A Markov Decision Process (MDP) provides a mathematical framework for sequential decision-making under uncertainty. It is formally defined as a 5-tuple:

$$(S, A, P, R, \gamma)$$

* **$S$**: State space (all possible states the agent can be in).
* **$A$**: Action space (all possible moves the agent can make).
* **$P(s' \mid s, a)$**: Transition probability function, defining the likelihood of landing in state $s'$ after taking action $a$ in state $s$.
* **$R(s, a)$**: Reward function, returning a scalar feedback signal from the environment.
* **$\gamma \in [0,1]$**: Discount factor, determining the present value of future rewards.

The defining characteristic of an MDP is the **Markov property**, which states that the future is conditionally independent of the past given the present state:

$$P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(S_{t+1} \mid S_t, A_t)$$

This implies that the current state $S_t$ is a completely sufficient summary of all historical interactions.

An agent's behavior is dictated by its **policy** ($\pi$):

* **Deterministic Policy:** $a = \pi(s)$
* **Stochastic Policy:** $\pi(a \mid s) = P(A_t = a \mid S_t = s)$

---

## 2. Return and Objective

The central objective in reinforcement learning is to maximize the expected cumulative reward over time, known as the **Return** ($G_t$):

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The discount factor $\gamma$ ensures that future rewards are weighted less than immediate gratification. When $\gamma < 1$, it mathematically guarantees the convergence of the infinite sum in continuing tasks.

The ultimate goal of an RL agent is to find an optimal policy $\pi^*$ that maximizes this expected return:

$$\pi^{*} = \arg\max_\pi \mathbb{E}_\pi [G_t]$$

Thus, reinforcement learning can be framed as an optimization problem over a policy space.

---

## 3. Value Functions and Bellman Equations

To evaluate the quality of policies, we define two types of value functions.

- **State-Value Function** ($$V^\pi(s)$$) : The expected return starting from state $s$ and following policy $\pi$ thereafter.


$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$


- **Action-Value Function** ($$Q^\pi(s,a)$$) : The expected return starting from state $s$, taking action $a$, and subsequently following policy $\pi$.

$$Q^\pi(s,a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$



These functions fulfill recursive structural equations called **Bellman Expectation Equations**, breaking down the value of a state (or state-action pair) into the immediate reward plus the discounted value of the succeeding state.

### 3.1 Derivation of the Bellman Expectation Equation for $V^\pi(s)$

We unpack the return $G_t$ into the immediate reward $R_{t+1}$ and the discounted next return $\gamma G_{t+1}$:

$$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

By the linearity of expectation, we separate the immediate reward from the future return:

$$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} \mid S_t = s] + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_t = s]$$

To calculate this expectation explicitly, we sum over all possible actions the agent might take via its policy $\pi(a \mid s)$, and all possible next states the environment might transition into via $P(s' \mid s, a)$:

$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \sum_{s' \in S} P(s' \mid s, a) \left[ R(s,a) + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] \right]$$

By definition,

$$
\mathbb{E}_\pi\!\left[G_{t+1}\mid S_{t+1}=s'\right]=V^\pi(s')
$$


Substituting this back yields the final recursive equation:

$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \sum_{s' \in S} P(s' \mid s, a) \left[ R(s,a) + \gamma V^\pi(s') \right]$$

### 3.2 Derivation of the Bellman Expectation Equation for $Q^\pi(s,a)$

Conditioning on both the initial state and action:

$$Q^\pi(s,a) = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a]$$

The immediate reward for taking action $a$ in state $s$ is expected as $R(s,a)$. The environment then moves to a next state $s'$ based on transition dynamics $P(s' \mid s, a)$:

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s']$$

We know that 

$$
\mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] = V^\pi(s')
$$

The value of that subsequent state $V^\pi(s')$ can be written in terms of $Q^\pi$ by averaging over all possible next actions $a'$ the policy could select:

$$V^\pi(s') = \sum_{a' \in A} \pi(a' \mid s') Q^\pi(s', a')$$

Substituting this back into the formulation for $Q^\pi(s,a)$ gives the recursive equation:

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \sum_{a' \in A} \pi(a' \mid s') Q^\pi(s', a')$$

---

## 4. Monte Carlo Methods

Monte Carlo (MC) methods learn value functions directly from **complete episodes** of experience. The actual empirical return $G_t$ is computed at the end of an episode:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$$

The value estimate for a state is updated incrementally using the prediction error (the difference between the target return and the current estimate):

$$V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))$$

where $\alpha$ is the learning rate.

### Characteristics of Monte Carlo:

* **Model-free:** No prior knowledge of transition probabilities or reward dynamics is required.
* **Unbiased:** Because it utilizes true, complete returns, the expected value of the estimator matches the true value.
* **High Variance:** Since a full episode contains many stochastic actions and state transitions, the cumulative variance can be massive.
* **Episodic Only:** It cannot be naturally deployed on continuing tasks that do not possess clear terminal boundaries.

---

## 5. Temporal Difference Learning and Q-Learning

Temporal Difference (TD) learning combines the model-free benefits of Monte Carlo methods with the iterative updates of Dynamic Programming. Unlike MC, TD methods do not wait for the end of an episode; they update estimates after a single step by **bootstrapping** off existing downstream estimates.

Q-learning is an off-policy TD control algorithm aimed at directly approximating the optimal action-value function $Q^{\ast}$, independent of the policy being executed by the agent. It is fundamentally anchored to the **Bellman Optimality Equation**:

$$Q^{\ast}(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q^{\ast}(S_{t+1}, a') \;\middle|\; S_t=s, A_t=a \right]$$

### 5.1 Intuition and Working of Q-Learning

Q-learning requires no structural model of the environment; it learns purely through trial and error. Consider an algorithmic analogy to error correction: if a computer model misclassifies a fruit, it receives negative feedback, adjusts its weights, and prevents that specific error in the future. Q-learning operates similarly by evaluating its action selections against realized environmental rewards.

#### The Q-Table

In tabular settings, Q-values are organized into a lookup grid called a **Q-table**:

* **Rows** represent every distinct state $s \in S$.
* **Columns** represent every distinct action $a \in A$.
* Each **Entry** stores a scalar value $Q(s,a)$, mapping out the agent’s current expectation of long-term reward for that pairing.

#### The Temporal Difference Update Rule

When an agent is in state $s$, executes action $a$, receives reward $r$, and arrives at next state $s'$, the Q-table entry is adjusted via:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]$$

* The term $\left[r + \gamma \max_{a'} Q(s',a')\right]$ represents the local **TD Target**.
* The difference between this target and the current estimate, $\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$, is the **TD Error**.

#### Exploration vs. Exploitation ($\epsilon$-Greedy Policy)

To guarantee convergence to the true optimal values, the agent must balance choosing what it knows to be profitable against exploring unknown alternatives. This is typically managed via an $\epsilon$-greedy policy:

* With probability $1 - \epsilon$: **Exploit** by selecting $\arg\max_a Q(s,a)$.
* With probability $\epsilon$: **Explore** by choosing a completely random action $a \in A$.

#### Step-by-Step Execution Loop

1. Initialize the Q-table entries arbitrarily (e.g., all zeros).
2. Observe the current state $s$.
3. Choose action $a$ using an $\epsilon$-greedy strategy.
4. Execute action $a$, then observe the reward $r$ and the new state $s'$.
5. Compute the TD error and update $Q(s,a)$ using the TD rule.
6. Set the state $s \leftarrow s'$.
7. Repeat steps 3–6 until a terminal state is reached, iterating across many distinct episodes.

## 5.2 From Optimal Q-Values to an Optimal Policy

Once the Q-values have successfully converged to the true optimal function $Q^{\ast}(s,a)$, extracting the optimal policy $\pi^{\ast}(s)$ is straightforward:

$$\pi^{\ast}(s) = \arg\max_a Q^{\ast}(s,a)$$

The equation indicates that the agent can behave optimally by performing a local, greedy evaluation over its actions at any state, choosing whichever action yields the highest expected long-term return.

#### Concrete Example

Suppose an agent reaches an intersection and evaluates its converged Q-table:

| Action ($a$) | $Q^*(s,a)$ |
| --- | --- |
| **Left** | 4 |
| **Right** | 12 |
| **Back** | 1 |

Applying the extraction rule:

$$\pi^*(s) = \arg\max_a [4, 12, 1] = \text{Right}$$

Knowing $Q^*(s,a)$ eliminates the need for deep planning; decision-making collapses into a direct table lookup.

---

## 6. Deep Q-Networks (DQN)

When confronting environments with massive or continuous state spaces like chess or raw screen pixels, maintaining a table becomes impossible. Deep Q-Networks solve this by substituting the table with a parameterized function approximator—a deep neural network:

$$Q(s,a) \approx Q(s,a;\theta)$$

where $\theta$ represents the weights of the network. The network is trained by minimizing a mean-squared Bellman error loss function at each iteration $i$.

Standard non-linear regression is very unstable in RL because consecutive states are highly correlated and targets change continuously. 

---

## 7. Policy Gradient and PPO

Rather than evaluating state configurations to infer a policy, **Policy Gradient** methods parameterize the policy directly as $\pi_\theta(a \mid s)$ and optimize it using **gradient ascent** to maximize expected total return:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$$

Using the mathematical identity known as the *likelihood ratio trick*, the gradient of this objective is evaluated without needing an explicit model of environmental transitions:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^\pi(s,a) \right]$$

### Proximal Policy Optimization (PPO)

Standard policy gradient steps frequently suffer from high step variance, causing the policy to degrade if a bad update is made. **Proximal Policy Optimization (PPO)** enforces stable learning updates by introducing a clipped surrogate objective function that penalizes changes that push the new policy too far from the old policy:

$$L^{PPO}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

Where the probability ratio is defined as:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

and $A_t$ is the **Advantage estimate** (measuring how much better an action is relative to the baseline state value). By clipping $r_t(\theta)$ within a small window $[1-\epsilon, 1+\epsilon]$, PPO ensures that step updates are incremental and stable.

---

## 8. Deep Deterministic Policy Gradient (DDPG)

DDPG is an off-policy **Actor-Critic** framework engineered specifically for environments with continuous action spaces (where finding $\max_a Q(s,a)$ over infinite options is intractable).

* **Actor Network ($\mu_\theta(s)$):** Deterministically maps a state directly to a specific continuous action output.
* **Critic Network ($Q_\phi(s,a)$):** Evaluates the chosen action, learning the standard action-value function via regression targets.

The actor's parameter weights are optimized by taking the gradient of the critic's output directly with respect to the action values, applying the chain rule:

$$\nabla_\theta J \approx \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_a Q_\phi(s,a) \big|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]$$

DDPG imports the experience replay and target network mechanisms from DQN to safely handle deep continuous optimization.

---

## 9. Soft Actor-Critic (SAC)

Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm tailored for continuous action spaces. It modifies the objective function by incorporating a secondary metric: **Entropy Maximization**. The objective function rewards the policy not just for gaining returns, but for acting as randomly and unpredictably as possible while doing so:

$$J(\pi) = \sum_{t} \mathbb{E}_{(S_t, A_t) \sim \rho_\pi} \left[ R(S_t, A_t) + \alpha \mathcal{H}(\pi(\cdot \mid S_t)) \right]$$

Where the local mathematical entropy is formulated as:

$$\mathcal{H}(\pi(\cdot \mid S_t)) = -\int_{A} \pi(a \mid S_t) \log \pi(a \mid S_t) \, da$$

The temperature parameter $\alpha$ regulates the trade-off between prioritizing raw reward and maximizing exploration entropy.

### Advantages of SAC:

* **Exploration:** By maximizing entropy, the agent searches more thoroughly through the action space, discovering alternative strategies that standard deterministic algorithms miss.
* **Robustness:** Because it learns a variety of viable paths to a goal rather than a single deterministic sequence, the policy adapts better to unexpected environmental perturbations.

---

## 10. Algorithm Comparison Matrix

| Algorithm | Type | Action Space | Policy Type | Core Mechanism / Differentiation |
| --- | --- | --- | --- | --- |
| **Monte Carlo** | Value-Based | Discrete / Continuous | On-Policy | Learns from full empirical returns at the end of complete episodes; high variance. |
| **Q-Learning** | Value-Based | Discrete | Off-Policy | Bootstraps values step-by-step via the tabular Bellman Optimality Equation. |
| **DQN** | Value-Based | Discrete | Off-Policy | Scales Q-learning to high dimensions using deep networks, experience replay, and target networks. |
| **PPO** | Policy Gradient | Discrete / Continuous | On-Policy | Direct policy optimization utilizing a clipped surrogate objective for step stability. |
| **DDPG** | Actor-Critic | Continuous | Off-Policy | Deterministic policy mapping tailored for smooth, continuous control problems. |
| **SAC** | Actor-Critic | Continuous | Off-Policy | Incorporates entropy maximization into its objective to enhance exploration and stability. |

---

## 11. Conclusion

Reinforcement learning can be formalized using Markov Decision Processes and solved via value-based or policy-based optimization structures. The Bellman expectation and optimality equations provide the mathematical backbone for these solutions, establishing that complex multi-step decision tasks can be broken down into recursive local updates.

While fundamental algorithms like Monte Carlo and Q-learning perform well in discrete, low-dimensional settings, modern deep reinforcement learning extends these concepts to complex environments. By utilizing deep networks alongside structural stabilizers—such as experience replay, target networks, clipped objectives, and entropy maximization—algorithms like DQN, PPO, DDPG, and SAC enable stable, model-free learning across high-dimensional and continuous control tasks.
