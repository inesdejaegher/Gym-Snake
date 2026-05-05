import glob
import os
import re
import pickle
import logging
import warnings
import gym
import gym_snake
import numpy as np

from helper_func import get_discrete_state, logbook_simulation

warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# -----------------------------
# ----- PARAMETERS TO SET -----
# -----------------------------
eval_episodes = 100
max_steps_without_consumption = 100


def extract_drug_reward(q_table_path):
    """
    Extracts the drug reward from a Q-table filename.

    Example:
    q_table_drug_reward_25_no_growth_EP_5000_TIME_04_05_2026_17-49-47.pkl
    returns 25.
    """
    q_table_name = os.path.basename(q_table_path)
    match = re.search(r"drug_reward_(\d+)_no_growth", q_table_name)

    if match is None:
        raise ValueError(f"Could not extract drug reward from: {q_table_name}")

    return int(match.group(1))


def find_q_tables(q_table_dir):
    """
    Finds all no-growth drug Q-tables and sorts them by drug reward.
    """
    pattern = os.path.join(q_table_dir, "q_table_drug_reward_*_no_growth_EP_5000_TIME_*.pkl")
    q_table_paths = glob.glob(pattern)
    return sorted(q_table_paths, key=extract_drug_reward)


def evaluate_q_table(q_table_path):
    """
    Evaluates one trained Q-table and saves one CSV file with evaluation results.
    """
    q_table_name = os.path.basename(q_table_path)
    drug_reward = extract_drug_reward(q_table_path)
    condition_name = f"drug_reward_{drug_reward}_no_growth"

    # ----- STORAGE FOLDER FOR RESULTS -----
    csv_name = f"Evaluation_Results_logbook_{q_table_name.replace('.pkl', '.csv')}"
    csv_dir = os.path.join(os.path.dirname(__file__), "..", "Results", "Drugs_No_Growth")
    os.makedirs(csv_dir, exist_ok=True)
    full_csv_path = os.path.join(csv_dir, csv_name)

    # Load the trained Q-table
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    logging.info(f"Loaded {condition_name} Q-table with {len(q_table)} known states.")

    # ------------------------------------------------------------------
    # ----- Initialize environment (MUST MATCH TRAINING SETTINGS!) -----
    # ------------------------------------------------------------------
    env = gym.make('snake-v0')
    base_env = env.unwrapped
    base_env.grid_size = [10, 10]
    base_env.n_foods = 1
    base_env.n_drugs = 1
    base_env.drug_reward = drug_reward
    base_env.drug_growth = 0

    # -----------------------------------
    # ----- Run the evaluation loop -----
    # -----------------------------------
    for episode in range(eval_episodes):
        env.reset()
        state = get_discrete_state(env)

        done = False
        total_reward = 0
        drugs_eaten_this_ep = 0
        food_eaten_this_ep = 0
        snake_length = 0
        steps = 0
        steps_without_consumption = 0
        loop = 0

        while not done:
            # --- PURE EXPLOITATION ---
            if state in q_table:
                action = int(np.argmax(q_table[state]))
            else:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)

            # ----- TRACK FOOD AND DRUGS EATEN DURING EPISODE -----
            ate_something = False
            
            # Track consumed drugs by looking at the info dictionary returned by the environment
            if info.get("drug_eaten", False):
                drugs_eaten_this_ep += 1
                ate_something = True

            # Track consumed food by checking the reward. (Subtract drug reward to isolate food reward)
            if reward - (base_env.drug_reward if info.get("drug_eaten", False) else 0) > 0:
                food_eaten_this_ep += 1
                ate_something = True

            if ate_something:
                steps_without_consumption = 0
            else:
                steps_without_consumption += 1

            # ----- TRACKING SNAKE LENGTH -----
            # Keep track of the snake's length.
            # We only update it if it's > 0 because on the final frame when the snake dies, 
            # the environment deletes the snake object entirely and returns a length of 0.
            if info.get("snake_length", 0) > 0:
                snake_length = info.get("snake_length", 0)

            # ----- MOVE TO NEXT STATE -----
            state = get_discrete_state(env)
            total_reward += reward
            steps += 1

            # ----- RENDER EVALUATION -----
            # Render the game so we can watch the trained agent
            # Delete if not needed
            # env.render()
            # time.sleep(0.05)  # Slow down the frames slightly to make it watchable

            # ----- LOOP PREVENTION -----
            if steps_without_consumption > max_steps_without_consumption:
                logging.info(f"{condition_name}: agent stuck in a loop. Forcing episode end.")
                loop = 1
                break

        # --- SAVE EPISODE RESULTS ---
        logbook_simulation(full_csv_path, episode, drugs_eaten_this_ep, food_eaten_this_ep, total_reward, snake_length, steps, loop)

        logging.info(
            f"{condition_name} | Evaluation Episode {episode + 1}/{eval_episodes} finished "
            f"| Total Reward: {total_reward} | Drugs: {drugs_eaten_this_ep} "
            f"| Food: {food_eaten_this_ep} | Snake Length: {snake_length} | Steps: {steps}"
        )

    env.close()
    logging.info(f"Evaluation complete for: {condition_name}")


if __name__ == "__main__":
    # Find all no-growth drug Q-tables in the specified directory
    q_table_dir = os.path.join(os.path.dirname(__file__), "..", "Q-Tables", "Drugs_No_Growth")
    q_table_paths = find_q_tables(q_table_dir)

    # Check if any Q-tables were found
    if len(q_table_paths) == 0:
        raise FileNotFoundError(f"No no-growth drug Q-tables found in: {q_table_dir}")

    # Evaluate each Q-table
    for q_table_path in q_table_paths:
        evaluate_q_table(q_table_path)
