# Week 4 Report: Temporal Difference Control

**Project:** Theoretical Foundations of RL (SoC 2026)

**Topic:** 
- On-Policy vs. Off-Policy Learning
- SARSA
- Q-Learning

**Resources Covered:** 
- Sutton and Barto (2020), Chapters 6 and 7
- David Silver's RL Course, Lecture 5 (Control)

---

## 1. Introduction to TD Control

While prediction focuses on estimating the value function $V(s)$ or $Q(s, a)$ for a fixed policy, **Control** aims to find the optimal policy. In model-free environments where transition probabilities $P_{ss'}^a$ and reward dynamics $R_s^a$ are completely unknown, the agent cannot look ahead to evaluate future states. It must explicitly maintain and learn the value of state-action pairs ($Q$-values) to guide behavior.

TD Control algorithms update these action-value estimates step-by-step using sampled experience, combining the model-free advantages of Monte Carlo with the bootstrapping efficiency of Dynamic Programming.

---

## 2. On-Policy vs. Off-Policy Learning

The framework of model-free control relies on the interplay between two distinct policy profiles:

* **The Behavioral Policy ($\mu$):** The strategy used by the agent to interact with the environment, select actions, and generate sample trajectories of experience.
* **The Target Policy ($\pi$):** The strategy that the agent evaluates, improves, and aims to optimize into the absolute best possible policy.

### On-Policy Learning

In on-policy learning, the target policy and the behavioral policy are identical ($\pi = \mu$). The agent learns about the policy it is actively executing. Because the agent must continuously explore to discover optimal paths, the policy being evaluated is typically stochastic (e.g., $\epsilon$-greedy). This means the value function converges to the value of a continuous, somewhat random exploration strategy rather than a pristine, optimal deterministic strategy.

### Off-Policy Learning

In off-policy learning, the target policy and the behavioral policy are decoupled ($\pi \neq \mu$). The agent follows an exploratory behavioral policy $\mu$ to collect raw experience data, but it updates its value estimates assuming it is strictly following a completely different target policy $\pi$. Typically, $\pi$ is set to be the 100% deterministic, greedy policy with respect to the current $Q$-values, allowing the agent to learn about the ideal optimal strategy while safely using a messy, chaotic strategy to explore the world.

---

## 3. SARSA: On-Policy TD Control

### 3.1 Mathematical Derivation

SARSA optimizes the action-value function by anchoring its updates directly to the **Bellman Expectation Equation for $Q$**:

$$Q^\pi(s, a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \sum_{a' \in A} \pi(a' \mid s') Q^\pi(s', a')$$

To perform model-free updates, we eliminate the structural transition dynamics and the policy summation by sampling a single sequence of experience from the active behavioral policy. This sequence consists of five sequential components: State ($S_t$), Action ($A_t$), Reward ($R_{t+1}$), Next State ($S_{t+1}$), and Next Action ($A_{t+1}$).

By using the value of the actual next action chosen by the policy, we form the **SARSA Target**:

$$\text{Target}_{\text{SARSA}} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$$

### 3.2 Algorithmic Update Rule

The value function shifts toward the target by a fractional learning step-size $\alpha$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right)$$

Because $A_{t+1}$ is sampled straight from the active exploratory policy, if the agent happens to select a highly dangerous exploratory action at step $t+1$, that negative outcome is reflected immediately in the update for $Q(S_t, A_t)$.

---

## 4. Q-Learning: Off-Policy TD Control

### 4.1 Mathematical Derivation

Q-Learning aims to bypass the current behavior entirely to estimate the absolute optimal action-value function $Q^{\ast}$. It derives its structural update from the **Bellman Optimality Equation for $Q$**:

$$Q^{\ast}(s, a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \max_{a'} Q^{\ast}(s', a')$$

When the agent samples a real-world transition, it transitions from $S_t$ via action $A_t$, receiving reward $R_{t+1}$ and landing in state $S_{t+1}$. Instead of looking at what action the behavioral policy *actually* picks next, Q-learning looks ahead and assumes that the target policy will act perfectly optimally from $S_{t+1}$ onward.

This constructs an off-policy target driven by a maximum operator:

$$\text{Target}_{\text{Q-Learning}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$$

### 4.2 Algorithmic Update Rule

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right)$$

By selecting $\max_{a'}$, the update isolates the value evaluation from the agent's exploratory deviations. The agent can take completely random actions to map out the environment, yet still successfully extract the exact mathematical structure of the optimal path.

---

## 5. Structural Comparison and Deliverables

### 5.1 Analytical Comparison Table

| Feature | SARSA | Q-Learning |
| --- | --- | --- |
| **Policy Regime** | On-Policy ($\pi = \mu$) | Off-Policy ($\pi \neq \mu$) |
| **Underlying Baseline** | Bellman Expectation Equation | Bellman Optimality Equation |
| **Lookahead Target** | Evaluates actual choice: $Q(S_{t+1}, A_{t+1})$ | Evaluates ideal choice: $\max_{a'} Q(S_{t+1}, a')$ |
| **Exploration Impact** | Vulnerable to exploratory penalties during training | Disregards exploratory penalties during updates |
| **Operational Profile** | Conservative and safe pathfinding | Aggressive and optimal pathfinding |

### 5.2 Behavioral Divergence: The Cliff Walking Problem

The classic demonstration of how these two update rules diverge is the **Cliff Walking** environment. Consider a gridworld where the shortest path to the goal runs directly along the edge of a deadly cliff. Falling off the cliff incurs a massive penalty (e.g., reward of $-100$) and resets the agent to the start.

* **SARSA Performance:** Because SARSA evaluates the *actual* actions it takes under its $\epsilon$-greedy behavior policy, it accounts for the fact that it occasionally takes random exploratory steps. It learns that walking right next to the cliff is dangerous because a random exploratory action could cause it to slip over the edge. Consequently, SARSA chooses a longer, safer path that stays multiple cells away from the cliff edge.
* **Q-Learning Performance:** Q-learning assumes perfect future optimization ($\max_{a'}$) regardless of exploration. It completely ignores the danger of its own random actions during updates. As a result, it learns the absolute shortest path right along the cliff edge. While this path is mathematically optimal under a perfect policy, during training the Q-learning agent will continuously fall off the cliff whenever a random exploratory step occurs.

---

## 6. Convergence Intuition for TD Control

To guarantee that model-free control algorithms converge cleanly to the true optimal action-value map ($Q \to Q^{\ast}$), specific conditions must be met regarding sampling density, step-sizes, and exploration decay.

### 6.1 The Robbins-Monro Conditions

The step-size sequence $\alpha_t$ must balance two competing statistical needs: it must be large enough to override arbitrary initialization biases, but small enough to eventually eliminate sampling noise. This is formalised by:

$$\sum_{t=1}^{\infty} \alpha_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty$$

* $\sum \alpha_t = \infty$ ensures that the step sizes do not decay too quickly, allowing the value function to travel across any distance from its initial starting state.
* $\sum \alpha_t^2 < \infty$ ensures that the steps eventually shrink down to zero, stabilizing the updates and preventing persistent oscillations around the true target values.

### 6.2 GLIE (Greedy in the Limit with Infinite Exploration)

For on-policy methods like SARSA to converge to the true optimal deterministic policy, the behavioral policy must slowly phase out its own randomness as training progresses. A control policy satisfies the GLIE properties if:

1. All state-action pairs are visited an infinite number of times (ensuring the agent never stops exploring unmapped paths).
2. The policy slowly collapses into a fully deterministic greedy policy as the number of time steps approaches infinity:

$$\lim_{t \to \infty} \epsilon_t = 0$$

Under GLIE conditions, SARSA’s exploratory policy gradually shifts from evaluating a noisy policy to evaluating the optimal policy, achieving convergence alongside Q-learning.

### 6.3 Q-Learning Convergence Space

Because Q-learning is structurally off-policy, its convergence proof is highly robust. It does not require a GLIE decay schedule to discover $Q^{\ast}$. The only strict prerequisite for Q-learning to converge to the optimal value map is that **every single state-action pair must be visited and updated an infinite number of times**. This ensures that no hidden high-value path is left permanently unexamined.