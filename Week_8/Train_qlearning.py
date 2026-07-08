import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()

print("Observation:", obs)
print("Observation Shape:", obs.shape)
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)