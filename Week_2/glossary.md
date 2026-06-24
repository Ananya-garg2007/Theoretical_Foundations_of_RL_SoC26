# Week 2: Reinforcement Learning Algorithms and Markov Decision Processes

## 1. Markov Decision Processes (MDPs)

A **Markov Decision Process (MDP)** is the fundamental mathematical framework used to describe reinforcement learning problems. It formalizes how an agent interacts with an environment over time.

An MDP is formally defined by the 5-tuple:
$$\left(S, A, P, R, \gamma\right)$$

* **$S$**: Set of all possible states
* **$A$**: Set of all possible actions
* **$P(s' \mid s, a)$**: Transition probability of moving to state $s'$ after taking action $a$ in state $s$
* **$R(s, a)$**: Reward received after taking action $a$ in state $s$
* **$\gamma$**: Discount factor ($0 \le \gamma \le 1$), determining the importance of future rewards

> 💡 **Intuition:** Think of playing a video game:
> 1. You are in a specific situation $\rightarrow$ **State**
> 2. You make a decision $\rightarrow$ **Action**
> 3. You receive feedback/points $\rightarrow$ **Reward**
> 4. The game moves to a new situation $\rightarrow$ **Next State**
> 
> This loop repeats indefinitely or until the game ends. An MDP is simply the mathematical language used to describe this loop.

### The Markov Property
$$P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots) = P(S_{t+1} \mid S_t, A_t)$$

* **Meaning:** The future depends *only* on the current state and action, not on the historic sequence of past states.
* **Why it matters:** This drastically simplifies learning. The agent does not need to store or remember its entire past history to make an optimal decision.

### Policy ($\pi$)
A policy defines the agent's behavior strategy:

* **Deterministic Policy:** Maps a state directly to a specific action.
    $$a = \pi(s)$$
* **Stochastic Policy:** Maps a state to a probability distribution over actions.
    $$\pi(a \mid s) = P(A_t = a \mid S_t = s)$$

---

## 2. Return and Objective

### Return ($G_t$)
The total accumulated discounted reward collected by the agent over time starting from time step $t$:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

> 💡 **Intuition:** > * Immediate rewards matter more than distant future rewards.
> * Future rewards are heavily discounted by $\gamma^k$ to model real-world uncertainty and mathematically prevent infinite sums in infinite-horizon tasks.

### Optimization Objective
The ultimate goal of the agent is to find an optimal policy $\pi^*$ that maximizes the expected return:
$$\pi^{*} = \arg\max_\pi \mathbb{E}_\pi [G_t]$$

---

## 3. Value Functions

Value functions quantify how favorable a specific state or action-state pair is for the agent.

### State-Value Function ($V^\pi(s)$)
The expected return starting from state $s$ and following policy $\pi$ thereafter:
$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

### Action-Value Function ($Q^\pi(s,a)$)
The expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter:
$$Q^\pi(s,a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

### Key Insights
| Function | Evaluation Target | Utility in RL |
| :--- | :--- | :--- |
| **$V(s)$** | Evaluates the overall goodness of **states**. | Useful for assessing environmental safety or progress. |
| **$Q(s,a)$** | Evaluates the goodness of specific **actions** within states. | Direct core of action selection; allows policy derivation without knowing environment transitions. |

---

## 4. Bellman Equations

The Bellman equations breakdown value functions recursively into an immediate reward and discounted downstream values.

### Bellman Expectation Equation for $V(s)$
$$V^\pi(s) = \mathbb{E}_\pi \big[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s \big]$$

> 💡 **Intuition:** The value of your current state is equal to the immediate reward you receive next, plus the discounted value of the state you land in.

#### Expanded Analytical Form
$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \sum_{s' \in S} P(s' \mid s, a) \big[ R(s, a) + \gamma V^\pi(s') \big]$$

### Bellman Expectation Equation for $Q(s, a)$
$$Q^\pi(s,a) = \mathbb{E}_\pi \big[ R_{t+1} + \gamma \mathbb{E}_{a' \sim \pi}[Q^\pi(S_{t+1}, a')] \big]$$

---

## 5. Derivation of the Bellman Equation

1. **Start with the definition of the state value function:**
   $$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

2. **Unroll the return definition ($G_t = R_{t+1} + \gamma G_{t+1}$):**
   $$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s]$$

3. **Factor out the discount factor ($\gamma$):**
   $$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \mid S_t = s]$$

4. **Substitute $G_{t+1}$ back into the equation:**
   $$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

5. **Apply the Law of Total Expectation to rewrite the return as a value function:**
   $$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s]$$

> 🛠️ **Core Takeaway:** This recursive restructuring allows us to break down long-term optimization problems into smaller iterative steps.

---

## 6. Monte Carlo Methods

Monte Carlo (MC) methods learn value functions directly from experiences gathered during **complete episodes**.

### Update Rule
$$V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))$$
*Where $\alpha$ is the learning rate.*

> 💡 **Intuition:** Play through the entire game/episode until it ends $\rightarrow$ see what your actual return ($G_t$) was $\rightarrow$ adjust your original estimates based on that final score.

### Properties
* **High Variance, Zero Bias:** Uses true empirical returns, but returns can fluctuate wildly based on stochastic choices.
* **Requirements:** Can only be applied to episodic environments that definitely terminate.

---

## 7. Temporal Difference (TD) Learning

TD Learning updates value estimates step-by-step at every time increment without waiting for an episode to end.

### Update Rule
$$V(S_t) \leftarrow V(S_t) + \alpha \big[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \big]$$

> 💡 **Intuition:** Learn step-by-step. Instead of waiting for the end of the game, adjust your estimate of your current state based immediately on the reward you got on the very next step along with your guess for the next state's value.

### Key Concept: Bootstrapping
TD learning relies on **bootstrapping**—updating an estimate based on another existing estimate rather than waiting for an absolute definitive result.

---

## 8. Q-Learning

Q-Learning is an **off-policy** Temporal Difference learning algorithm that explicitly seeks to discover the optimal action-value function $Q^*$.

### Update Rule
$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ R + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]$$

### The Exploration vs. Exploitation Dilemma
* **Exploitation:** Choosing the action currently known to yield the highest expected reward.
* **Exploration:** Trying unknown or unoptimized actions to gather more data.

### $\epsilon$-Greedy Strategy
To balance this tradeoff, agents choose actions via an $\epsilon$-greedy scheme:
$$\text{Action} = \begin{cases} \text{Random action} & \text{with probability } \epsilon \\ \arg\max_a Q(s,a) & \text{with probability } 1-\epsilon \end{cases}$$

---

## 9. From Q-Values to Policy

Once the optimal action-value function ($Q^*$) is learned, extracting the optimal deterministic policy ($\pi^*$) becomes straightforward:

$$\pi^{*}(s) = \arg\max_a Q^{*}(s,a)$$

> 💡 **Meaning:** Look at the current state, review the expected returns for all available actions in your Q-table, and pick the one with the highest value.

---

## 10. Deep Q-Networks (DQN)

Tabular Q-learning fails when state-action spaces grow too massive to fit into memory matrices (e.g., Chess, Go, or raw screen pixels).

### Core Idea
Replace the discrete Q-table with a parameterized **Neural Network Function Approximator**:
$$Q(s,a) \approx Q(s,a; \theta)$$

### Stabilizing Techniques
1.  **Experience Replay:** Stores experience transitions $(s, a, r, s')$ in a memory buffer and samples them randomly. This breaks temporal correlation between sequential data samples.
2.  **Target Networks:** Uses a separate neural network ($\theta^-$) to compute target values that is updated only periodically, avoiding chasing a moving training target.

---

## 11. Policy Gradient and PPO

Instead of estimating value functions to indirectly find policies, **Policy Gradient** methods explicitly parameterize and optimize the policy network $\pi_\theta$.

### Objective Function
$$J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$$
We perform gradient ascent ($\nabla_\theta J(\theta)$) to increase the probability of actions that yield higher positive returns.

### Proximal Policy Optimization (PPO)
Standard policy gradients suffer from destructively large updates. PPO introduces a clipped objective function to constrain step size:

$$L^{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

* **$r_t(\theta)$**: Probability ratio of the new policy vs. old policy.
* **$\hat{A}_t$**: Advantage estimate (how much better an action is compared to average).
* **Clipping:** Prevents the policy from changing too drastically in a single update step, ensuring stable training.

---

## 12. Deep Deterministic Policy Gradient (DDPG)

DDPG is an Actor-Critic algorithm designed specifically for continuous action spaces (e.g., robotic joint manipulation, steering wheel angles).

### Architecture
* **Actor Network:** Dictates actions given states ($\mu(s;\theta^\mu)$).
* **Critic Network:** Evaluates action choices by outputting $Q(s,a)$.

> 💡 **Intuition:** One network acts (the Actor), while the other network critiques how good that action was to guide optimization (the Critic).

---

## 13. Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic method that incorporates **entropy regularized optimization**.

### Objective Function
$$J(\pi) = \sum_{t=1}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \big[ R(s_t, a_t) + \alpha H(\pi(\cdot \mid s_t)) \big]$$

* **$H(\pi)$**: Entropy, representing the randomness of the policy.
* **$\alpha$**: Temperature parameter controlling the trade-off between reward optimization and entropy.

> 💡 **Intuition:** SAC rewards the agent for acting randomly and unpredictably when rewards are equal. This explicitly incentivizes broad environment exploration and prevents premature convergence to sub-optimal local minima.

---

## Conclusion: The Grand RL Loop

Every method covered in Reinforcement Learning follows the exact same underlying cycle:

┌────────────────────────┐
│  Estimate Values /     │
│  Evaluate Current State│
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Improve Policy /      │
│  Optimize Decisions    │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Explore Environment / │
│  Collect New Rewards   │
└───────────┬────────────┘
            │
            └───────────────────────┘


```markdown
* **MDPs** give us the math equations.
* **Bellman Equations** break problems down recursively.
* **MC/TD Learning** helps us learn from real environment experiences.
* **Deep Learning (DQN, PPO, SAC)** scales these methods up to solve highly complex, real-world problems.