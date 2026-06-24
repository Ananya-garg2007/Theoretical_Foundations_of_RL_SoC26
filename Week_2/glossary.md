## Glossary of Key Terms

* **Markov Property:** A property of a stochastic process where future states depend exclusively on the current state and action, making historical trajectories irrelevant for prediction.
* **Bootstrapping:** The process of updating a value estimate using other, subsequent learned value estimates rather than waiting for an absolute empirical return.
* **Model-Free:** RL algorithms that learn behaviors via direct interaction with the environment without constructing or utilizing explicit transition probabilities $P(s' \mid s, a)$ or reward functions.
* **On-Policy vs. Off-Policy:** *On-policy* methods optimize the exact policy that is currently being executed to collect experience (e.g., PPO). *Off-policy* methods evaluate and optimize a target policy while collecting data using a separate behavior policy (e.g., Q-learning, DQN, SAC).
* **Experience Replay:** A memory buffer that records historical agent experiences $(s, a, r, s')$, allowing deep networks to sample transitions randomly to break up temporal correlation.
* **Actor-Critic:** An algorithmic structure split into two entities: an **Actor** that maps states to action distributions, and a **Critic** that evaluates those actions by estimating state values or action values.
* **Entropy Maximization:** An optimization technique that penalizes predictable behavior by rewarding policies for maintaining diversity in their action choices, preventing premature convergence to local optima.
* **Advantage ($A(s,a)$):** A structural metric defined as $Q(s,a) - V(s)$, quantifying how much better a specific action is compared to the average expected return of that state.

---
