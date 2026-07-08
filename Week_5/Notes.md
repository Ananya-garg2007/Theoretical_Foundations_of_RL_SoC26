# Week 5: Value Function Approximation and Deep Q-Networks

## 1. Why Function Approximation?

In classical reinforcement learning algorithms such as tabular Q-learning, the agent stores a separate Q-value for every possible state-action pair in a Q-table. This approach works well when the number of states and actions is relatively small. However, in many real-world problems such as robotics, autonomous driving, or Atari games, the state space becomes extremely large or even continuous. In these situations, maintaining a Q-table becomes computationally impossible because the number of state-action pairs grows exponentially.

To overcome this limitation, reinforcement learning uses function approximation. Instead of storing a value for every state-action pair, the agent learns a function that estimates these values.

Instead of learning $Q(s,a)$ for every individual state-action pair, we learn:

$$Q(s,a;\theta)$$

where $\theta$ represents the parameters of the approximating function.

Thus, the learning problem changes from storing values to learning the parameters that best approximate the true value function.

---

## 2. Linear Function Approximation

The simplest form of function approximation assumes that the value function is a linear combination of manually designed features.

Suppose each state is represented by a feature vector $\phi(s)$. Then the value function is approximated as:

$$V(s;\theta) = \theta^T \phi(s)$$

Similarly, the action-value function becomes:

$$Q(s,a;\theta) = \theta^T \phi(s,a)$$

where:
* **$\theta$** is the weight vector.
* **$\phi(s,a)$** is the feature vector describing the state-action pair.

The objective of learning is to update the weights $\theta$ so that the predicted values become as close as possible to the true expected returns.

**Advantages:**
* Computationally efficient.
* Easy to interpret.
* Strong theoretical guarantees.

**Limitations:**
* Linear approximators assume that the relationship between features and value is linear. Most real-world environments are highly nonlinear, making linear models insufficient.

---

## 3. Nonlinear Function Approximation

Modern reinforcement learning replaces linear approximators with deep neural networks.

Instead of $Q(s,a) = \theta^T \phi(s,a)$, we learn:

$$Q(s,a;\theta) = f_\theta(s,a)$$

where $f_\theta$ is a deep neural network.

Rather than relying on handcrafted features, the neural network automatically learns useful representations directly from raw observations such as images or sensor data. This allows reinforcement learning algorithms to solve much more complex problems than classical methods.

**Advantages:**
* Learns complex nonlinear relationships.
* Handles very high-dimensional state spaces.
* Eliminates manual feature engineering.

**Limitations:**
* Training neural networks introduces instability because the learning targets themselves keep changing.

---

## 4. Deep Q-Networks (DQN)

Deep Q-Networks (DQN) combine the Bellman equation from Q-learning with the representational power of deep neural networks. Instead of maintaining a Q-table, a neural network predicts the Q-values corresponding to each action.

For a given state, $Q(s,a;\theta)$ is the predicted value produced by the neural network. The desired target value $y$ is obtained using the Bellman optimality equation:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

where:
* **$r$** is the immediate reward.
* **$\gamma$** is the discount factor.
* **$\theta^-$** denotes the parameters of the target network.

The network is trained by minimizing the mean squared error between the predicted Q-value and the target. The DQN loss function is therefore:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left(y - Q(s,a;\theta)\right)^2 \right]$$

or equivalently:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a; \theta) \right)^2 \right]$$

The parameters $\theta$ are updated using gradient descent to minimize this loss.

---

## 4.5 From Bellman Equation to the DQN Loss Function

The foundation of Q-learning is the Bellman Optimality Equation, which states that the optimal value of a state-action pair is the expected immediate reward plus the discounted maximum expected value of the next state:

$$
Q^*(s,a)
=
\mathbb{E}\left[
R_{t+1}
+
\gamma
\max_{a'} Q^*(S_{t+1},a')
\mid
S_t=s,\; A_t=a
\right]
$$

In classical Q-learning, we use this recursive relationship to update our Q-values iteratively. The update rule pushes our current estimate $Q(s,a)$ towards the **Temporal Difference (TD) target**:

$$\text{Target} = R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$$

When transitioning to Deep Q-Networks, we replace the tabular Q-values with a neural network $Q(s,a; \theta)$. Instead of iterative assignment, we treat the TD target as the "ground truth" (albeit a moving one) for our network to predict.

Using the target network parameters $\theta^-$ to stabilize this ground truth, we define our specific scalar target for a sampled transition $(s, a, r, s')$ as:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

To train the main network (with parameters $\theta$), we frame this as a standard supervised learning regression problem. We want to minimize the difference between our current prediction and the TD target. We use the Mean Squared Error (MSE) loss function over a mini-batch of transitions sampled from the replay buffer $D$:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a; \theta) \right)^2 \right]$$

Taking the gradient of this loss function with respect to the weights $\theta$ allows us to use stochastic gradient descent (SGD) to optimize the network, explicitly tying the theoretical Bellman equation to the backpropagation process.

---

## 5. Why Does Naive DQN Become Unstable?

Replacing a Q-table with a neural network introduces two major problems.

### Correlated Samples
Traditional gradient descent assumes that training examples are approximately independent. However, an RL agent collects experiences sequentially ($State_1 \rightarrow State_2 \rightarrow State_3 \rightarrow \dots$). 

These observations are highly correlated. Training directly on such correlated samples causes unstable parameter updates and poor convergence.

### Moving Targets
The target value is computed using the same network that is being updated. As the parameters change, $Q(s,a;\theta)$ also changes. Consequently, the target continuously shifts during optimization. This creates a situation where the network is attempting to chase a moving target, often causing oscillations or divergence.

---

## 6. Experience Replay

To overcome correlated data, DQN stores each interaction in a replay buffer. Each experience is stored as a transition tuple:

$$(s, a, r, s')$$

Instead of training on consecutive transitions, the algorithm randomly samples mini-batches from this replay memory. Random sampling decorrelates the training data and allows each experience to be reused multiple times.

**Benefits:**
* Breaks correlations between samples.
* Improves sample efficiency.
* Produces smoother gradient updates.
* Reduces variance.

---

## 7. Target Networks

DQN introduces a second neural network called the target network. There are now two networks:

### Online Network
* **Parameters:** $\theta$
* This network is updated after every gradient step.

### Target Network
* **Parameters:** $\theta^-$
* This network remains fixed for many iterations and is periodically updated by copying the weights of the online network.

The target therefore becomes:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

Since $\theta^-$ changes much more slowly than $\theta$, the optimization target becomes nearly stationary.

**Benefits:**
* Stabilizes training.
* Prevents oscillations.
* Improves convergence.

---

## 8. Why Replay Buffers and Target Networks Work Together

Experience replay addresses one source of instability by ensuring that training data are approximately independent and identically distributed (i.i.d.). Target networks address another source by preventing the Bellman target from changing too rapidly.

Together, these two ideas transformed Q-learning from an unstable algorithm into a practical deep reinforcement learning algorithm capable of learning directly from raw visual input. These innovations were the key contributions of the original DeepMind DQN (Nature, 2015) paper.

---

## Conclusion

Function approximation enables reinforcement learning to scale from simple tabular problems to complex environments with millions of possible states. Linear approximators provide a mathematically simple solution but struggle to model complex relationships. Deep neural networks overcome this limitation by learning nonlinear value functions directly from data.

Deep Q-Networks combine classical Q-learning with deep learning, replacing the Q-table with a neural network. The addition of experience replay and target networks addresses the instability introduced by nonlinear function approximation, making DQN one of the foundational algorithms in modern deep reinforcement learning.