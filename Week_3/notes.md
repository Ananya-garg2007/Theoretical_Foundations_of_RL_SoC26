# Week 3: Dynamic Programming & Model-Free Prediction

**Project:** Theoretical Foundations of RL (SoC 2026)

**Topic:** 
- Bellman expectation and optimality equations
- Policy evaluation, policy iteration, value iteration
- Convergence intuition
- Monte Carlo vs TD learning
- TD(0), TD(lambda)
- Bias-variance tradeoff

**Resources Covered:** 
- David Silver Lectures 3-4
- Sutton and Barto (2020), Ch. 4-6

---

## 1. The Bellman Equations: Expectation vs. Optimality

In Reinforcement Learning, the value of a state or a state-action pair can be decomposed recursively into an immediate reward plus the discounted expected value of the succeeding state.

### 1.1 The Environment Model

A full Model-Based environment is formalized as a Markov Decision Process (MDP) tuple $\langle S, A, P, R, \gamma \rangle$, where:

* $P_{ss'}^a = P(S_{t+1} = s' \mid S_t = s, A_t = a)$ defines the transition dynamics.
* $R_s^a = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$ defines the expected immediate reward.

---

### 1.2 The Bellman Expectation Equation

The Expectation equation answers the question: *"If I am stuck acting according to my current, messy policy $\pi$, what is the long-term value of my current state?"* It represents a linear system of equations across the state space.

#### Mathematical Derivation:

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

$$V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

By applying the law of total expectation to condition on the action taken and the subsequent state transitioned to:


$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V^\pi(s') \right)$$

* **The Outer Sum ($\sum_a \pi(a \mid s)$):** This accounts for your internal policy randomness (e.g., if you choose between actions via a probability distribution).
* **The Inner Sum ($\sum_{s'} P_{ss'}^a$):** This accounts for the environment's structural transition randomness.
* **Linearity:** Because there are no operators like $\max$ or $\min$, this is a standard system of linear equations. If you have $N$ states, you can solve this exactly using matrix inversion ($V = (I - \gamma P)^{-1} R$).

Similarly, for the state-action value function $Q^\pi(s, a)$:


$$Q^\pi(s, a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \sum_{a' \in A} \pi(a' \mid s') Q^\pi(s', a')$$

---

### 1.3 The Bellman Optimality Equation

The Optimality equation answers the question: *"If I discard my current strategy and act flawlessly from this moment onward, what is the maximum possible value of this state?"* Because the optimal policy must choose the action that maximizes the expected return, this equation is fundamentally non-linear due to the inclusion of the $\max$ operator.

#### State Value Function $V^{\ast}(s)$:

$$V^{\ast}(s) = \max_{a \in A} Q^{\ast}(s, a)$$

$$V^{\ast}(s) = \max_{a \in A} \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V^{\ast}(s') \right)$$

#### Action Value Function $Q^{\ast}(s, a)$:

$$Q^{\ast}(s, a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \max_{a' \in A} Q^{\ast}(s', a')$$

* **The $\max_a$ Operator:** We no longer care about the policy's distribution $\pi(a \mid s)$. We assume the agent will look at all available choices and deliberately pick the single action that yields the highest expected return.
* **Non-Linearity:** The $\max$ operator makes it impossible to solve using standard linear algebra matrix inversion. We are forced to use iterative optimization methods to let the numbers gradually settle into place.

---

## 2. Dynamic Programming Algorithms

Dynamic Programming (DP) uses the Bellman equations as iterative update rules. To guarantee convergence, updates are performed synchronously across the entire state space during each iteration sweep.

### 2.1 Iterative Policy Evaluation

Iterative Policy Evaluation computes the state-value function $V^\pi$ for an arbitrary policy $\pi$. It iteratively applies the Bellman Expectation Equation, turning it into an assignment rule.


```text
Input:
    Policy π

Initialize:
    V(s) ← 0, ∀ s ∈ S
    V(terminal) ← 0
    θ > 0

Repeat:
    Δ ← 0

    For each state s ∈ S:
        v ← V(s)

        V(s) ← ∑ₐ π(a | s)
                ∑ₛ′ P(s′ | s, a)
                [R(s, a, s′) + γV(s′)]

        Δ ← max(Δ, |v − V(s)|)

Until Δ < θ

Return V ≈ V^π
```


---

### 2.2 Policy Iteration

Policy Iteration systematically improves performance by alternating between two discrete phases:

1. **Policy Evaluation:** Computing $V^\pi$ to true convergence under the current policy $\pi$.
2. **Policy Improvement:** Generating a strictly better or equal policy $\pi' \geq \pi$ by acting greedily with respect to $V^\pi$.

```text
1. Initialization

Initialize:
    V(s) ← 0, ∀ s ∈ S
    π(s) ← arbitrary action, ∀ s ∈ S

--------------------------------------------------

2. Policy Evaluation

Repeat:
    Δ ← 0

    For each state s ∈ S:
        v ← V(s)

        V(s) ← ∑ₛ′ P(s′ | s, π(s))
                [R(s, π(s), s′) + γV(s′)]

        Δ ← max(Δ, |v − V(s)|)

Until Δ < θ

--------------------------------------------------

3. Policy Improvement

policy_stable ← true

For each state s ∈ S:
    old_action ← π(s)

    π(s) ← arg maxₐ
            ∑ₛ′ P(s′ | s, a)
            [R(s, a, s′) + γV(s′)]

    If old_action ≠ π(s):
        policy_stable ← false

If policy_stable:
    Return V, π
Else:
    Go to Step 2
```

#### The Policy Improvement Theorem:

The logical foundation of this algorithm rests on the proof that a greedy policy update guarantees monotonic improvement:


$$Q^\pi(s, \pi'(s)) = \max_{a \in A} Q^\pi(s, a) \geq Q^\pi(s, \pi(s)) = V^\pi(s)$$


By inductive unrolling of the inequality, it holds that $V^{\pi'}(s) \geq V^\pi(s)$ for all $s \in S$.

---

### 2.3 Value Iteration

Value Iteration collapses the evaluation and improvement loop into a single step by updating the value function directly via the Bellman Optimality Equation. It skips explicit policy tracking completely until convergence is achieved.

```text
Initialize V(s) = 0 for all s in S, and V(terminal) = 0
Parameter: \theta > 0

Loop:
  \Delta \leftarrow 0
  For each s in S:
    v \leftarrow V(s)
    V(s) \leftarrow \max_{a} \sum_{s'} P_{ss'}^a [ R_{ss'}^a + \gamma V(s') ]
    \Delta \leftarrow \max(\Delta, |v - V(s)|)
until \Delta < \theta

Output a deterministic optimal policy \pi \approx \pi^{\ast} such that:
\pi(s) = \arg\max_{a} \sum_{s'} P_{ss'}^a [ R_{ss'}^a + \gamma V(s') ]

```

---

## 3. Convergence Intuition: Contraction Mappings

The convergence of Dynamic Programming algorithms is mathematically anchored to the **Banach Fixed-Point Theorem** via operators in a complete metric space (a Banach space under the infinity norm).

Let $\mathcal{T}^\pi$ be the Bellman Expectation operator:


$$\mathcal{T}^\pi V(s) = \sum_{a \in A} \pi(a \mid s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V(s') \right)$$

An operator $\mathcal{T}$ is a $\gamma$-contraction if the distance between any two value functions $U$ and $V$ shrinks by at least $\gamma$ after applying the operator:


$$\|\mathcal{T} U - \mathcal{T} V\|_\infty \leq \gamma \|U - V\|_\infty$$

Where the infinity norm $\|\cdot\|_\infty$ selects the maximum absolute difference across all states:


$$\|U - V\|_\infty = \max_{s \in S} |U(s) - V(s)|$$

### Proof Intuition for Contraction:

$$\mathcal{T} U(s) - \mathcal{T} V(s) = \gamma \sum_{s'} P_{ss'}^a (U(s') - V(s'))$$

$$|\mathcal{T} U(s) - \mathcal{T} V(s)| \leq \gamma \sum_{s'} P_{ss'}^a |U(s') - V(s')| \leq \gamma \sum_{s'} P_{ss'}^a \|U - V\|_\infty = \gamma \|U - V\|_\infty$$

Because $\gamma < 1$, repeatedly applying $\mathcal{T}$ continuously reduces the distance between the current function and the unique optimal solution. This guarantees that no matter how you initialize your value function, the algorithm will always converge to the exact same fixed point.

---

## 4. Model-Free Prediction: MC vs. TD

When the environment matrices $P$ and $R$ are unknown, the agent must drop planning updates and switch to learning from raw trajectories of experience.

### 4.1 Monte Carlo (MC) Methods

Monte Carlo methods learn value functions directly from completed episodes of experience. The target value for an update is the actual empirical total return $G_t$ observed at the end of the episode.

#### Update Equation:

$$V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)$$


Where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T$.

---

### 4.2 Temporal Difference Learning (TD(0))

Temporal Difference learning updates value estimates online after every single state transition. Instead of waiting for the terminal state to compute $G_t$, it builds an immediate estimate called the **TD Target** by combining the observed reward with its own downstream guess.

#### Update Equation:

$$V(S_t) \leftarrow V(S_t) + \alpha \left( \text{TD Target} - V(S_t) \right)$$

$$V(S_t) \leftarrow V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right)$$

The term $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD Error**.

---

### 4.3 The Bias-Variance Tradeoff

The fundamental cleavage between MC and TD is rooted in the classical statistical bias-variance tradeoff:

* **Monte Carlo (Unbiased, High Variance):**
* **Bias:** Precisely zero ($\mathbb{E}[G_t] = V^\pi(S_t)$). It does not rely on intermediate assumptions or secondary guesses.
* **Variance:** High. The return $G_t$ accumulates the compounding randomness of every single action choice and state transition along the entire trajectory.


* **Temporal Difference (Biased, Low Variance):**
* **Bias:** High early on. Because TD uses the current, unconverged value function $V(S_{t+1})$ to construct its target, it updates its estimates based on another guess (**bootstrapping**).
* **Variance:** Low. The target only spans a single step of structural environmental randomness ($R_{t+1}$ and $S_{t+1}$), shielding the update from downstream volatility.



---

### 4.4 TD($\lambda$): Unifying the Spectrum

Rather than choosing between updating across 1 step ($TD(0)$) or an infinite horizon ($MC$), **TD($\lambda$)** unifies the two regimes by calculating a geometric average of all available $n$-step returns using a decay parameter $\lambda \in [0, 1]$.

#### The $\lambda$-Return (Forward View):

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

* When $\lambda = 0$, $G_t^\lambda$ collapses exactly into the 1-step target of $TD(0)$.
* When $\lambda = 1$, $G_t^\lambda$ expands exactly into the full trajectory return of standard Monte Carlo.

---

## 5. Mathematical Walkthrough: Toy MDP Comparison

We evaluate a completely deterministic, model-free environment to isolate and analyze how MC and TD(0) update mechanisms behave under highly restricted data profiles.

### 5.1 Environment Configuration

* **State Space:** Non-terminal states $S_0, S_1$, and terminal state $S_T$.
* **Dynamics:** * Transition 1: $S_0 \rightarrow S_1$ with reward $R_1 = +2$
* Transition 2: $S_1 \rightarrow S_T$ with reward $R_2 = +10$


* **Hyperparameters:** Discount factor $\gamma = 1.0$, learning rate $\alpha = 0.5$.
* **Initialization:** Initial value parameters are uniform zeroes: $V(S_0) = 0, \quad V(S_1) = 0, \quad V(S_T) = 0$.

```text
[ S_0 ] --(R=+2)--> [ S_1 ] --(R=+10)--> [ S_T (Terminal) ]

```

The agent records exactly one sample episode through the environment:


$$\text{Trajectory:} \quad S_0 \xrightarrow{R_1=2} S_1 \xrightarrow{R_2=10} S_T$$

---

### 5.2 Scenario A: Monte Carlo Execution

Monte Carlo calculation delays all changes until the absolute end of the complete trajectory sequence.

#### 1. Empirical Return Calculation

* For state $S_1$:

$$G_1 = R_2 = 10$$


* For state $S_0$:

$$G_0 = R_1 + \gamma R_2 = 2 + 1.0(10) = 12$$



#### 2. Value Optimization Step

$$V(S_0) \leftarrow V(S_0) + \alpha (G_0 - V(S_0)) = 0 + 0.5(12 - 0) = 6$$

$$V(S_1) \leftarrow V(S_1) + \alpha (G_1 - V(S_1)) = 0 + 0.5(10 - 0) = 5$$

$$\text{Post-MC State Values:} \quad V(S_0) = 6, \quad V(S_1) = 5$$

---

### 5.3 Scenario B: Temporal Difference TD(0) Execution

TD(0) evaluates changes sequentially, modifying values immediately as each state transition occurs.

#### 1. Processing Step 1 ($S_0 \rightarrow S_1$)

* **Observed Reward:** $R_1 = 2$
* **Target Calculation:** $R_1 + \gamma V(S_1) = 2 + 1.0(0) = 2$
* **Update Execution:**

$$V(S_0) \leftarrow V(S_0) + \alpha \left( R_1 + \gamma V(S_1) - V(S_0) \right)$$


$$V(S_0) \leftarrow 0 + 0.5(2 + 0 - 0) = 1$$



#### 2. Processing Step 2 ($S_1 \rightarrow S_T$)

* **Observed Reward:** $R_2 = 10$
* **Target Calculation:** $R_2 + \gamma V(S_T) = 10 + 1.0(0) = 10$
* **Update Execution:**

$$V(S_1) \leftarrow V(S_1) + \alpha \left( R_2 + \gamma V(S_T) - V(S_1) \right)$$


$$V(S_1) \leftarrow 0 + 0.5(10 + 0 - 0) = 5$$



$$\text{Post-TD(0) State Values:} \quad V(S_0) = 1, \quad V(S_1) = 5$$

---

### 5.4 Comparative Analysis of Results

| State | Initial Value | True $V^\pi(s)$ | MC Post-Value | TD(0) Post-Value |
| --- | --- | --- | --- | --- |
| **$S_0$** | $0$ | $12$ | **$6$** | **$1$** |
| **$S_1$** | $0$ | $10$ | **$5$** | **$5$** |

#### Why did $V(S_0)$ diverge so drastically?

* **The Bootstrapping Phenomenon:** When TD(0) processed the transition from $S_0$ to $S_1$, it looked forward at the initialized value of $S_1$ ($V(S_1) = 0$). Because it relied on this uninformative downstream guess, the update to $S_0$ was severely dragged down toward zero. This clearly demonstrates the mathematical manifestation of **TD Bias**.
* **The Information Delay:** In TD(0), information about the large reward ($+10$) at the end of the episode only managed to move upstream by exactly one step (into $S_1$). It will require a second complete episode for that updated value of $S_1$ ($5$) to propagate backward and properly boost the value of $S_0$.
* **The MC Difference:** Because Monte Carlo waited for the trajectory to terminate, it was able to use the actual, complete historical return ($12$) to update $S_0$ immediately. This shows why MC handles poor value initializations much more effectively in deterministic environments.

```

```