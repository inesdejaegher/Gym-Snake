import numpy as np
np.bool8 = np.bool_
import gym
import gym_snake
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time

# ----------------------------
# Direction constants
# ----------------------------

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
    food_color = np.array([0, 0, 255], dtype=np.uint8)

    foods = []
    grid_w, grid_h = map(int, grid_object.grid_size)

    for x in range(grid_w):
        for y in range(grid_h):
            if grid_object.food_space((x, y)):
                foods.append(np.array([x, y]))

    return foods

def get_drug_positions(env):
    # drug_positions comes from your modified SnakeEnv
    return [np.array(p) for p in env.unwrapped.drug_positions]

def get_snake_length(env):
    controller = env.unwrapped.controller
    if controller is None:
        return 0

    snake = controller.snakes[0]
    if snake is None:
        snake = controller.dead_snakes[0]

    if snake is None:
        return 0

    return 1 + len(snake.body)

def manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

def cell_is_blocked(controller, pos):
    grid_w, grid_h = map(int, controller.grid.grid_size)
    x, y = int(pos[0]), int(pos[1])

    # wall
    if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
        return True

    # snake collision
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
    State features:
    - danger straight / left / right
    - nearest normal food direction
    - nearest drug direction
    - current direction
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

    # nearest normal food
    foods = get_food_positions(controller.grid)
    if len(foods) == 0:
        food = head
    else:
        food = min(foods, key=lambda f: manhattan(head, f))

    food_left = int(food[0] < head[0])
    food_right = int(food[0] > head[0])
    food_up = int(food[1] < head[1])
    food_down = int(food[1] > head[1])

    # nearest drug
    drugs = get_drug_positions(env)
    if len(drugs) == 0:
        drug = head
    else:
        drug = min(drugs, key=lambda d: manhattan(head, d))

    drug_left = int(drug[0] < head[0])
    drug_right = int(drug[0] > head[0])
    drug_up = int(drug[1] < head[1])
    drug_down = int(drug[1] > head[1])

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
        drug_left,
        drug_right,
        drug_up,
        drug_down,
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
# Create env
# ----------------------------

env = gym.make("snake-v0")
base_env = env.unwrapped
base_env.grid_size = [10, 10]
base_env.n_foods = 2
base_env.n_drugs = 1
base_env.snake_size = 4

# ----------------------------
# Q-learning containers
# ----------------------------

q_table = defaultdict(lambda: np.zeros(4))
episode_rewards = []
episode_lengths = []
episode_drug_pickups = []
episode_food_like_rewards = []
episode_snake_lengths = []
episode_snake_growth = []

epsilon = EPSILON

# ----------------------------
# Training loop
# ----------------------------

for episode in range(EPISODES):
    obs = env.reset()
    start_snake_length = get_snake_length(env)
    state = get_state(env)

    if state is None:
        episode_rewards.append(0)
        episode_lengths.append(0)
        episode_drug_pickups.append(0)
        episode_food_like_rewards.append(0)
        episode_snake_lengths.append(start_snake_length)
        episode_snake_growth.append(0)
        continue

    total_reward = 0
    steps = 0
    done = False
    drug_pickups = 0
    food_rewards = 0

    while not done and steps < MAX_STEPS:
        action = choose_action(q_table, state, epsilon)
        obs, reward, done, info = env.step(action)

        # bookkeeping: infer what happened from reward
        if reward >= 6:
            drug_pickups += 1
        elif reward == 1:
            food_rewards += 1

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
    episode_drug_pickups.append(drug_pickups)
    episode_food_like_rewards.append(food_rewards)
    final_snake_length = get_snake_length(env)
    episode_snake_lengths.append(final_snake_length)
    episode_snake_growth.append(final_snake_length - start_snake_length)

    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        avg_length = np.mean(episode_lengths[-50:])
        avg_drugs = np.mean(episode_drug_pickups[-50:])
        avg_foods = np.mean(episode_food_like_rewards[-50:])
        avg_snake_length = np.mean(episode_snake_lengths[-50:])
        avg_snake_growth = np.mean(episode_snake_growth[-50:])
        print(
            f"Episode {episode + 1:4d} | "
            f"avg reward: {avg_reward:.3f} | "
            f"avg length: {avg_length:.1f} | "
            f"avg drugs: {avg_drugs:.2f} | "
            f"avg foods: {avg_foods:.2f} | "
            f"avg snake length: {avg_snake_length:.2f} | "
            f"avg growth: {avg_snake_growth:.2f} | "
            f"epsilon: {epsilon:.3f}"
        )

env.close()

# ----------------------------
# Plot training progress
# ----------------------------

window = 25

def moving_avg(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return out

smoothed_rewards = moving_avg(episode_rewards, window)
smoothed_lengths = moving_avg(episode_lengths, window)
smoothed_drugs = moving_avg(episode_drug_pickups, window)
smoothed_foods = moving_avg(episode_food_like_rewards, window)

plt.figure()
plt.plot(episode_rewards, label="Episode reward")
plt.plot(smoothed_rewards, label="Smoothed reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning with food + drug")
plt.legend()
plt.show()

plt.figure()
plt.plot(episode_lengths, label="Episode length")
plt.plot(smoothed_lengths, label="Smoothed length")
plt.xlabel("Episode")
plt.ylabel("Steps survived")
plt.title("Episode length")
plt.legend()
plt.show()

plt.figure()
plt.plot(episode_drug_pickups, label="Drug pickups")
plt.plot(smoothed_drugs, label="Smoothed drug pickups")
plt.xlabel("Episode")
plt.ylabel("Drug pickups")
plt.title("Drug-seeking behaviour")
plt.legend()
plt.show()

plt.figure()
plt.plot(episode_food_like_rewards, label="Normal food pickups")
plt.plot(smoothed_foods, label="Smoothed normal food pickups")
plt.xlabel("Episode")
plt.ylabel("Food pickups")
plt.title("Healthy food-seeking behaviour")
plt.legend()
plt.show()

# ----------------------------
# Watch trained snake
# ----------------------------

env = gym.make("snake-v0")
base_env = env.unwrapped
base_env.grid_size = [10, 10]
base_env.n_foods = 2
base_env.n_drugs = 1
base_env.snake_size = 4

obs = env.reset()
state = get_state(env)
done = False

print("Watching trained snake...")
print("Grid:", base_env.controller.grid.grid_size)
print("Foods:", base_env.n_foods)
print("Drugs:", base_env.n_drugs)

watch_total_reward = 0
watch_drugs = 0
watch_foods = 0
watch_start_snake_length = get_snake_length(env)

if state is not None:
    while not done:
        env.render()
        action = int(np.argmax(q_table[state]))
        obs, reward, done, info = env.step(action)

        if reward >= 6:
            watch_drugs += 1
            print(f"Drug taken | reward={reward}")
        elif reward == 1:
            watch_foods += 1
            print(f"Food taken | reward={reward}")

        watch_total_reward += reward

        if not done:
            state = get_state(env)
            if state is None:
                break

        time.sleep(0.15)

env.close()

print("Watch summary:")
print("Total reward:", watch_total_reward)
print("Drug pickups:", watch_drugs)
print("Food pickups:", watch_foods)
watch_final_snake_length = get_snake_length(env)
print("Final snake length:", watch_final_snake_length)
print("Snake growth:", watch_final_snake_length - watch_start_snake_length)
