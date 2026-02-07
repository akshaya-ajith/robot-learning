import gymnasium as gym
import panda_gym
import time

env = gym.make('PandaPickAndPlace-v3', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.05)

    if terminated or truncated:
        observation, info = env.reset()