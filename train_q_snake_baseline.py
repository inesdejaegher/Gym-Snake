import gym
import gym_snake
import numpy as np
np.bool8 = np.bool_
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time

# ----------------------------
# Direction constants
# ----------------------------

#Comment: ik heb net ontdekt dat er onder gym_snake.envs.snake allemaal .py bestanden zitten die de env maken en grotendeel hiervan dus al hebbebn geïmplementeerd, 
#dus kan wrs veel korter door die functies gwn te importeren.

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# ----------------------------
# Helpers
# ----------------------------

def turn_left(direction):
    return (direction - 1) % 4

def turn_right(direction):
    return (direction + 1) % 4

def next_pos(pos, direction):
    x, y = int(pos[0]), int(pos[1])

    if direction == UP:
        return np.array([x, y - 1])
    if direction == RIGHT:
        return np.array([x + 1, y])
    if direction == DOWN:
        return np.array([x, y + 1])
    if direction == LEFT:
        return np.array([x - 1, y])

    raise ValueError(f"Invalid direction: {direction}")

def get_food_positions(grid_object):
    """
    Infer food positions from the rendered grid pixels.
    Food color in this repo is [0, 0, 255].
    """
    food_color = np.array([0, 0, 255], dtype=np.uint8)

    grid_pixels = grid_object.grid
    h, w = grid_pixels.shape[:2]

    unit = int(grid_object.unit_size)
    gap = int(grid_object.unit_gap)
    step = unit + gap

    foods = []

    for py in range(0, h, step):
        for px in range(0, w, step):
            pixel = grid_pixels[py, px]
            if np.array_equal(pixel, food_color):
                gx = px // step
                gy = py // step
                foods.append(np.array([gx, gy]))

    return foods

def cell_is_blocked(controller, pos):
    """
    Returns True if the position hits a wall or any snake segment.
    """
    grid_w, grid_h = map(int, controller.grid.grid_size)
    x, y = int(pos[0]), int(pos[1])

    # Wall collision
    if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
        return True

    # Snake collision
    for snake in controller.snakes:
        if snake is None:
            continue

        if np.array_equal(snake.head, pos):
            return True

        for body_part in snake.body:
            if np.array_equal(body_part, pos):
                return True

    return False

def get_state(env):
    """
    Build a compact discrete state for Q-learning.
    Returns None if the snake is dead / missing.
    """
    controller = env.unwrapped.controller
    snake = controller.snakes[0]

    if snake is None:
        return None

    head = snake.head.copy()
    direction = int(snake.direction)

    straight_dir = direction
    left_dir = turn_left(direction)
    right_dir = turn_right(direction)

    pos_straight = next_pos(head, straight_dir)
    pos_left = next_pos(head, left_dir)
    pos_right = next_pos(head, right_dir)

    danger_straight = int(cell_is_blocked(controller, pos_straight))
    danger_left = int(cell_is_blocked(controller, pos_left))
    danger_right = int(cell_is_blocked(controller, pos_right))

    foods = get_food_positions(controller.grid)
    if len(foods) == 0:
        food = head
    else:
        food = foods[0]

    food_left = int(food[0] < head[0])
    food_right = int(food[0] > head[0])
    food_up = int(food[1] < head[1])
    food_down = int(food[1] > head[1])

    dir_up = int(direction == UP)
    dir_right = int(direction == RIGHT)
    dir_down = int(direction == DOWN)
    dir_left = int(direction == LEFT)

    return (
        danger_straight,
        danger_left,
        danger_right,
        food_left,
        food_right,
        food_up,
        food_down,
        dir_up,
        dir_right,
        dir_down,
        dir_left,
    )

def choose_action(q_table, state, epsilon):
    if state is None:
        return random.randint(0, 3)

    if random.random() < epsilon:
        return random.randint(0, 3)

    return int(np.argmax(q_table[state]))

# ----------------------------
# Training settings
# ----------------------------

EPISODES = 3000
MAX_STEPS = 300
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

# ----------------------------
# Train
# ----------------------------
env = gym.make("snake-v0")
base_env = env.unwrapped
base_env.grid_size = [10, 10]
base_env.n_foods = 3
base_env.snake_size = 4

q_table = defaultdict(lambda: np.zeros(4))
episode_rewards = []
episode_lengths = []

epsilon = EPSILON

for episode in range(EPISODES):
    obs = env.reset()


    state = get_state(env)
    if state is None:
        episode_rewards.append(0)
        episode_lengths.append(0)
        continue

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < MAX_STEPS:
        action = choose_action(q_table, state, epsilon)
        obs, reward, done, info = env.step(action)

        next_state = get_state(env)

        if done or next_state is None:
            target = reward
            done = True
        else:
            target = reward + GAMMA * np.max(q_table[next_state])

        q_table[state][action] += ALPHA * (target - q_table[state][action])

        if not done and next_state is not None:
            state = next_state

        total_reward += reward
        steps += 1

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        avg_length = np.mean(episode_lengths[-50:])
        print(
            f"Episode {episode + 1:4d} | "
            f"avg reward (last 50): {avg_reward:.3f} | "
            f"avg length (last 50): {avg_length:.1f} | "
            f"epsilon: {epsilon:.3f}"
        )

env.close()

# ----------------------------
# Plot training progress
# ----------------------------

window = 20
smoothed = []

for i in range(len(episode_rewards)):
    start = max(0, i - window + 1)
    smoothed.append(np.mean(episode_rewards[start:i + 1]))

plt.figure()
plt.plot(episode_rewards, label="Episode reward")
plt.plot(smoothed, label="Smoothed reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning on snake-v0")
plt.legend()
plt.show()

# ----------------------------
# Watch trained snake
# ----------------------------

env = gym.make("snake-v0")
base_env = env.unwrapped
base_env.grid_size = [10, 10]
base_env.n_foods = 3
base_env.snake_size = 4

obs = env.reset()

print("Watch grid:", base_env.controller.grid.grid_size)
print("Watch foods:", len(get_food_positions(base_env.controller.grid)))

state = get_state(env)
done = False

if state is not None:
    while not done:
        env.render()
        action = int(np.argmax(q_table[state]))
        obs, reward, done, info = env.step(action)

        if not done:
            state = get_state(env)
            if state is None:
                break

        time.sleep(0.15)

env.close()