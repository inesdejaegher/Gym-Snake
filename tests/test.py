import numpy as np
np.bool8 = np.bool_
import gym
import gym_snake
import time

env = gym.make("snake-v0")
base_env = env.unwrapped

base_env.grid_size = [10, 10]
base_env.n_foods = 2
base_env.n_drugs = 1
base_env.snake_size = 4

obs = env.reset()

for _ in range(200):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("reward:", reward)
    time.sleep(0.1)

    if done:
        obs = env.reset()

env.close()