import gym
import gym_snake
import time

env = gym.make("snake-v0")
base_env = env.unwrapped

# Change the ENV SETTINGS before reset
base_env.grid_size = [10, 10]
base_env.n_foods = 3
base_env.snake_size = 4

obs = env.reset()

print("Configured grid_size:", base_env.grid_size)
print("Actual grid_size after reset:", base_env.controller.grid.grid_size)
print("Configured n_foods:", base_env.n_foods)
print("Actual number of snakes:", len(base_env.controller.snakes))

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    time.sleep(0.1)
    if done:
        obs = env.reset()

env.close()