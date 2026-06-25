# Week 1: Introduction to Reinforcement Learning

**Project:** Theoretical Foundations of RL (SoC 2026)

**Topic:** 
- Agent-environment interaction, return, policies, value functions
- Exploration vs exploitation (high-level intuition)
- Formalizing RL problems as optimization

**Resources Covered:** 
- David Silver Lecture 1: https://youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-&si=f71naScldk1Mbfwi
- Sutton and Barto (2020), Ch. 1: http://incompleteideas.net/book/RLbook2020.pdf
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

---

## Introduction

Reinforcement Learning refers to a process that is designed to make decisions through interaction with the environment in such a way as to maximize the total reward.

Unlike supervised learning, n RL there is no need for labeled inputs/output, instead, the agent learns from interaction or trial and error.

Two key characteristics:

**Trial and-error learning**:  
The agent is not told what the correct action is. Instead, it tries different actions, observes the outcomes, and gradually improves its behavior.
The agent tries actions, observes outcomes, and improves over time.

**Delayed reward**:  
In many situations, the effect of an action is not immediate. An action taken now might only lead to a reward much later. Because of this, the agent must learn to associate its actions with long-term consequences rather than just immediate gains.

These two aspects make reinforcement learning fundamentally different and more challenging than standard learning settings.

---

## Agent–Environment Interaction

The basic structure of reinforcement learning is an interaction loop between the agent and the environment.

1. Observe state  
2. Take action  
3. Receive reward  
4. Move to next state  

The state can be thought of as a compact summary of everything important from the past that the agent needs to make a decision.

Over time, this interaction generates a sequence of states, actions, and rewards, which together form the history, that is, the complete record of everything the agent has experienced so far.

History: Hₜ = (S₁, A₁, R₂, S₂, A₂, R₃, …, Sₜ)

A well-defined state satisfies the **Markov property**, meaning that the future depends only on the current state and not on the full history.

---

## Return

In reinforcement learning, the objective is to maximize the return, which is the total accumulated reward over time. Instead of focusing only on immediate rewards, the agent considers the entire sequence of future rewards.

To model this, future rewards are usually discounted, meaning rewards received sooner are considered more valuable than those received later. This helps the agent balance short-term and long-term gains.

---

## Policies

A policy defines the behavior of the agent by specifying what action to take in each state. Basically a mapping from states to actions.

- Deterministic policy: the agent always chooses the same action in a given state (a = π(s))
- Stochastic policy: the agent chooses actions with certain probabilities (π(a|s) = P[A = a | S = s])  

At the beginning, the policy may be random because the agent does not yet understand the environment. But as learning progresses, the policy improves and becomes more focused on actions that lead to higher rewards.

Goal: find optimal policy maximizing expected return.

---

## Value Functions

Value functions help the agent evaluate how good a state or an action is in terms of expected future rewards.

Instead of just looking at immediate rewards, value functions estimate how much reward the agent can expect in the long run.

- Value of state V(s): tells us how good it is to be in that state
- value of a state-action pair Q(s,a): tells us how good it is to take a particular action in that state

This helps the agent compare different choices and make better decisions.

---

## Exploration vs Exploitation

A key challenge in reinforcement learning is balancing exploration and exploitation.

Exploitation means choosing actions that are already known to give good rewards. Exploration means trying new actions to discover whether they might lead to even better outcomes.

- **Exploitation**: use known good actions  
- **Exploration**: try new actions  

If the agent only exploits, it may miss better strategies. If it only explores, it may never settle on a good strategy. Therefore, a balance between the two is necessary for effective learning.

---

## RL as Optimization

Reinforcement learning can be viewed as an optimization problem where the goal is to find a policy that maximizes expected return.

This is typically modeled using a framework called a Markov Decision Process, which describes the states, actions, rewards, and how the system evolves over time.

Solving a reinforcement learning problem involves estimating how good different actions are, improving the policy based on this, and repeating the process until a good strategy is found.

---

## Conclusion

Reinforcement learning provides a powerful framework for learning through interaction. By combining trial-and-error learning with the challenge of delayed rewards, it enables agents to make decisions that optimize long-term outcomes.

Instead of being told what is correct, the agent learns from experience, gradually improving its behavior while balancing exploration and exploitation. These ideas form the foundation of reinforcement learning and make it both challenging and widely applicable.

---

## Intuition

RL is like learning strategies in real life:

- Try different approaches  
- Learn what works  
- Improve over time  

Example: choosing between your favorite restaurant (exploit) vs trying a new one (explore).