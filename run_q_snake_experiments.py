import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import gym
import gym_snake
import numpy as np

```python run_q_snake_experiments.py \
  --episodes 3000 \
  --seeds 0,1,2 \
  --n-drugs 0,1,2 \
  --drug-growths 0,3,6 \
  --drug-rewards 6
````

np.bool8 = np.bool_

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


@dataclass
class ExperimentConfig:
    run_name: str
    seed: int
    episodes: int
    max_steps: int
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    grid_width: int
    grid_height: int
    snake_size: int
    n_foods: int
    n_drugs: int
    drug_reward: int
    drug_growth: int


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
    foods = []
    grid_w, grid_h = map(int, grid_object.grid_size)

    for x in range(grid_w):
        for y in range(grid_h):
            if grid_object.food_space((x, y)):
                foods.append(np.array([x, y]))

    return foods


def get_drug_positions(env):
    return [np.array(p) for p in env.unwrapped.drug_positions]


def get_snake_length(env):
    return env.unwrapped.get_snake_length()


def manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def cell_is_blocked(controller, pos):
    grid_w, grid_h = map(int, controller.grid.grid_size)
    x, y = int(pos[0]), int(pos[1])

    if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
        return True

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
    food = head if len(foods) == 0 else min(foods, key=lambda f: manhattan(head, f))

    drugs = get_drug_positions(env)
    drug = head if len(drugs) == 0 else min(drugs, key=lambda d: manhattan(head, d))

    return (
        danger_straight,
        danger_left,
        danger_right,
        int(food[0] < head[0]),
        int(food[0] > head[0]),
        int(food[1] < head[1]),
        int(food[1] > head[1]),
        int(drug[0] < head[0]),
        int(drug[0] > head[0]),
        int(drug[1] < head[1]),
        int(drug[1] > head[1]),
        int(direction == UP),
        int(direction == RIGHT),
        int(direction == DOWN),
        int(direction == LEFT),
    )


def choose_action(q_table, state, epsilon):
    if state is None or random.random() < epsilon:
        return random.randint(0, 3)

    return int(np.argmax(q_table[state]))


def make_env(config):
    env = gym.make("snake-v0")
    base_env = env.unwrapped
    base_env.grid_size = [config.grid_width, config.grid_height]
    base_env.n_foods = config.n_foods
    base_env.n_drugs = config.n_drugs
    base_env.snake_size = config.snake_size
    base_env.drug_reward = config.drug_reward
    base_env.drug_growth = config.drug_growth
    return env


def run_experiment(config, run_dir):
    random.seed(config.seed)
    np.random.seed(config.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    env = make_env(config)
    q_table = defaultdict(lambda: np.zeros(4))
    epsilon = config.epsilon
    rows = []

    for episode in range(config.episodes):
        env.reset()
        start_snake_length = get_snake_length(env)
        state = get_state(env)

        total_reward = 0
        steps = 0
        done = False
        drug_pickups = 0
        food_pickups = 0

        while not done and steps < config.max_steps and state is not None:
            action = choose_action(q_table, state, epsilon)
            _, reward, done, info = env.step(action)

            drug_eaten = bool(info.get("drug_eaten", False))
            controller_reward = reward - (config.drug_reward if drug_eaten else 0)

            if drug_eaten:
                drug_pickups += 1
            if controller_reward == 1:
                food_pickups += 1

            next_state = get_state(env)

            if done or next_state is None:
                target = reward
                done = True
            else:
                target = reward + config.gamma * np.max(q_table[next_state])

            q_table[state][action] += config.alpha * (target - q_table[state][action])

            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        final_snake_length = get_snake_length(env)

        rows.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "steps": steps,
            "drug_pickups": drug_pickups,
            "food_pickups": food_pickups,
            "start_snake_length": start_snake_length,
            "final_snake_length": final_snake_length,
            "snake_growth": final_snake_length - start_snake_length,
            "epsilon": epsilon,
        })

        if (episode + 1) % 50 == 0:
            recent = rows[-50:]
            avg_reward = np.mean([row["total_reward"] for row in recent])
            avg_growth = np.mean([row["snake_growth"] for row in recent])
            print(
                f"{config.run_name} | episode {episode + 1:4d} | "
                f"avg reward: {avg_reward:.2f} | avg growth: {avg_growth:.2f}"
            )

    env.close()

    episode_path = run_dir / "episodes.csv"
    with episode_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(config, rows)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def summarize(config, rows):
    last_window = rows[-min(50, len(rows)):]

    def mean(key, values):
        return float(np.mean([row[key] for row in values]))

    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "episodes": config.episodes,
        "parameters": asdict(config),
        "overall": {
            "avg_reward": mean("total_reward", rows),
            "avg_steps": mean("steps", rows),
            "avg_drug_pickups": mean("drug_pickups", rows),
            "avg_food_pickups": mean("food_pickups", rows),
            "avg_final_snake_length": mean("final_snake_length", rows),
            "avg_snake_growth": mean("snake_growth", rows),
        },
        "last_50": {
            "avg_reward": mean("total_reward", last_window),
            "avg_steps": mean("steps", last_window),
            "avg_drug_pickups": mean("drug_pickups", last_window),
            "avg_food_pickups": mean("food_pickups", last_window),
            "avg_final_snake_length": mean("final_snake_length", last_window),
            "avg_snake_growth": mean("snake_growth", last_window),
        },
    }


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_grid_size(value):
    parts = parse_int_list(value)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid size must look like 10,10")
    return parts


def build_configs(args):
    grid_width, grid_height = args.grid_size
    configs = []

    for seed, n_drugs, drug_reward, drug_growth, snake_size in product(
        args.seeds,
        args.n_drugs,
        args.drug_rewards,
        args.drug_growths,
        args.snake_sizes,
    ):
        run_name = (
            f"seed-{seed}_drugs-{n_drugs}_reward-{drug_reward}_"
            f"growth-{drug_growth}_snake-{snake_size}"
        )
        configs.append(ExperimentConfig(
            run_name=run_name,
            seed=seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            grid_width=grid_width,
            grid_height=grid_height,
            snake_size=snake_size,
            n_foods=args.n_foods,
            n_drugs=n_drugs,
            drug_reward=drug_reward,
            drug_growth=drug_growth,
        ))

    return configs


def save_summary_csv(summaries, path):
    rows = []
    for summary in summaries:
        params = summary["parameters"]
        row = {
            "run_name": summary["run_name"],
            "seed": summary["seed"],
            "n_drugs": params["n_drugs"],
            "drug_reward": params["drug_reward"],
            "drug_growth": params["drug_growth"],
            "snake_size": params["snake_size"],
            "last_50_avg_reward": summary["last_50"]["avg_reward"],
            "last_50_avg_steps": summary["last_50"]["avg_steps"],
            "last_50_avg_drug_pickups": summary["last_50"]["avg_drug_pickups"],
            "last_50_avg_food_pickups": summary["last_50"]["avg_food_pickups"],
            "last_50_avg_final_snake_length": summary["last_50"]["avg_final_snake_length"],
            "last_50_avg_snake_growth": summary["last_50"]["avg_snake_growth"],
        }
        rows.append(row)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Q-learning snake addiction simulations and save structured results."
    )
    parser.add_argument("--out", default="results/q_snake_addiction")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seeds", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--grid-size", type=parse_grid_size, default=[10, 10])
    parser.add_argument("--snake-sizes", type=parse_int_list, default=[4])
    parser.add_argument("--n-foods", type=int, default=2)
    parser.add_argument("--n-drugs", type=parse_int_list, default=[0, 1, 2])
    parser.add_argument("--drug-rewards", type=parse_int_list, default=[6])
    parser.add_argument("--drug-growths", type=parse_int_list, default=[0, 3, 6])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out) / timestamp
    configs = build_configs(args)
    summaries = []

    print(f"Running {len(configs)} simulations")
    print(f"Saving results to {output_dir}")

    for config in configs:
        run_dir = output_dir / config.run_name
        summaries.append(run_experiment(config, run_dir))

    save_summary_csv(summaries, output_dir / "summary.csv")

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Done. Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
