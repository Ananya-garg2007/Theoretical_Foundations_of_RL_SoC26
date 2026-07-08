# Week 6: Policy Gradient Methods

## 1. Why Policy Gradient Methods?

In previous weeks, we studied value-based methods such as Q-learning and DQN, where the goal was to estimate the value of each state or state-action pair and then derive a policy from these values.

However, value-based methods have certain limitations. They struggle with continuous action spaces, and the process of first estimating values and then extracting a policy may not always be efficient.

Policy gradient methods take a fundamentally different approach. Instead of learning a value function and then deriving a policy, they directly optimize the policy itself. Suppose the policy is parameterized by $\theta$:

$$\pi_\theta(a|s)$$

The objective is to find the parameters $\theta$ that maximize the expected cumulative reward.

## 2. Objective Function

The performance of a policy is measured by its expected return:

$$J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$$

where:

* $\tau$ denotes a trajectory.
* $R(\tau)$ is the total reward accumulated along the trajectory.

The optimization problem becomes:

$$\theta^* = \arg\max_\theta J(\theta)$$

Unlike Q-learning, where we optimize value functions, policy gradient methods directly optimize this objective.

## 3. Deriving the Policy Gradient Theorem

The objective can be written as:

$$J(\theta) = \sum_\tau P(\tau;\theta) R(\tau)$$

where $P(\tau;\theta)$ is the probability of generating trajectory $\tau$ under policy $\pi_\theta$.

Taking the gradient:

$$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau;\theta) R(\tau)$$

Using the **log-derivative trick**:

$$\nabla_\theta P(\tau) = P(\tau) \nabla_\theta \log P(\tau)$$

we obtain:

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau) \nabla_\theta \log P(\tau) R(\tau)$$

which becomes:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log P(\tau) R(\tau) \right]$$

Since the environment dynamics do not depend on $\theta$:

$$P(\tau) = \rho(s_0) \prod_t P(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t)$$

Taking logarithms:

$$\log P(\tau) = \log\rho(s_0) + \sum_t \log P(s_{t+1}|s_t,a_t) + \sum_t \log\pi_\theta(a_t|s_t)$$

Differentiating, all environment terms vanish because they are independent of $\theta$. Therefore:

$$\nabla_\theta \log P(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Substituting this back:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$$

This is the **Policy Gradient Theorem**. The theorem is remarkable because it shows that we never need to differentiate through the environment. We only differentiate the policy.

## 4. The REINFORCE Algorithm

REINFORCE is the simplest Monte Carlo policy gradient algorithm. Instead of using the total trajectory reward, it uses the return from each timestep:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t}R_{k+1}$$

The gradient estimate becomes:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]$$

The parameters are updated by gradient ascent:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

Unlike Q-learning, REINFORCE waits until an episode finishes before updating the policy.

## 5. Why Does REINFORCE Have High Variance?

Although REINFORCE provides an unbiased estimate of the policy gradient, its variance is often very large. This happens because:

* Returns can vary significantly across episodes.
* The same action may receive very different rewards.
* Updates depend on complete trajectories.

Large variance causes noisy gradient estimates, making learning unstable and slow.

## 6. Variance Reduction Using a Baseline

A key observation is that subtracting a quantity that does not depend on the action does not change the expected gradient. Thus:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t-b(s_t)) \right]$$

where $b(s)$ is called the **baseline**. The most common choice is the state-value function:

$$b(s) = V^\pi(s)$$

The quantity:

$$A(s,a) = Q(s,a) - V(s)$$

is called the **advantage function**. It measures how much better an action is than the average action in that state. Using advantages:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) A(s_t,a_t) \right]$$

This significantly reduces variance while keeping the estimator unbiased.

## 7. REINFORCE with Baseline

The complete algorithm is:

1. Generate one complete episode using the current policy.
2. Compute returns $G_t$.
3. Estimate the baseline $V(s_t)$.
4. Compute the advantage $A_t = G_t - V(s_t)$.
5. Update parameters:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) A_t$$


6. Repeat until convergence.

## 8. Why Does the Baseline Reduce Variance?

Suppose two identical actions are taken in the same state. Without a baseline:

* Reward 120
* Reward 118

These produce very different gradients. With a baseline of 119, the updates become $+1$ and $-1$ respectively. The average gradient remains unchanged, but the magnitude of fluctuations is greatly reduced.

Hence:

* The estimator remains unbiased.
* Gradient estimates become smoother.
* Convergence becomes faster.
* Training becomes significantly more stable.

## Conclusion

Policy gradient methods optimize policies directly instead of estimating value functions. The Policy Gradient Theorem shows that the gradient of the expected return depends only on the policy and not on the environment dynamics. REINFORCE applies this theorem using Monte Carlo returns, while introducing a baseline transforms the algorithm into a much more stable estimator by reducing variance without introducing bias.

