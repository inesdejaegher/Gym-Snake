import glob
import os
import re
import pickle
import logging
import warnings
import gym
import gym_snake
import numpy as np
import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


from helper_func import get_discrete_state, logbook_simulation

warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# -----------------------------
# ----- PARAMETERS TO SET -----
# -----------------------------
eval_episodes = 500
# max_steps_without_consumption = 100


def extract_drug_reward(q_table_path):
    """
    Extracts the drug reward from a Q-table filename.

    Example:
    "DRUG_R10_ENG_NO_PEN_G1_Q_EP15K_TIME_XXX.pkl"
    returns 10.
    """
    q_table_name = os.path.basename(q_table_path)
    match = re.search(r"DRUG_R(\d+)_ENG_NO_PEN_G1", q_table_name)

    if match is None:
        raise ValueError(f"Could not extract drug reward from: {q_table_name}")

    return int(match.group(1))


def find_q_tables(q_table_dir):
    """
    Finds all energy no penalty drug Q-tables and sorts them by drug reward.
    """
    pattern = os.path.join(q_table_dir, "DRUG_R*_ENG_NO_PEN_G1_Q_EP*_TIME_*.pkl")
    q_table_paths = glob.glob(pattern)
    return sorted(q_table_paths, key=extract_drug_reward)


def evaluate_q_table(q_table_path):
    """
    Evaluates one trained Q-table and saves one CSV file with evaluation results.
    """
    q_table_name = os.path.basename(q_table_path)
    drug_reward = extract_drug_reward(q_table_path)
    condition_name = f"DRUG_R{drug_reward}_ENG_NO_PEN_G1"

    # ----- STORAGE FOLDER FOR RESULTS -----
    base_name = q_table_name.split("TIME_")[0] + "TIME"
    csv_name = f"EVAL_{base_name}_{datetime.datetime.now().strftime('%d_%m_%Y_%H-%M-%S')}.csv"
    csv_dir = os.path.join(os.path.dirname(__file__), "..", "Results", "Drugs_Energy_No_Penalty_G1")
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
    base_env.max_energy = 100
    base_env.step_energy_cost = 1
    base_env.drug_resets_energy = True
    base_env.drug_energy_penalty_factor = 0

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
        last_known_energy = 100


        while not done:
            # --- PURE EXPLOITATION ---
            if state in q_table:
                action = int(np.argmax(q_table[state]))
            else:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)

            # Grab the snake object
            snake = env.unwrapped.controller.snakes[0] 
    
            # Only update our tracker if the snake actually exists in memory this frame
            if snake is not None:
                last_known_energy = snake.energy

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

            # ----- LOOP PREVENTION: NOT NEEDED WITH THE NEW ENERGY TRACKER -----
            # if steps_without_consumption > max_steps_without_consumption:
            #     logging.info(f"{condition_name}: agent stuck in a loop. Forcing episode end.")
            #     loop = 1
            #     break

        # --- SAVE EPISODE RESULTS ---
        logbook_simulation(full_csv_path, episode, drugs_eaten_this_ep, food_eaten_this_ep, total_reward, snake_length, steps, loop)

        logging.info(
            f"{condition_name} | Evaluation Episode {episode + 1}/{eval_episodes} finished "
            f"| Total Reward: {total_reward} | Drugs: {drugs_eaten_this_ep} "
            f"| Food: {food_eaten_this_ep} | Snake Length: {snake_length} | Steps: {steps}"
            f"| Final Energy: {last_known_energy}"
        )

    env.close()
    logging.info(f"Evaluation complete for: {condition_name}")
    return full_csv_path


if __name__ == "__main__":
    # Find all no-energy-penalty drug Q-tables in the specified directory
    q_table_dir = os.path.join(os.path.dirname(__file__), "..", "Q-Tables", "Drugs_Energy_No_Penalty_G1")
    q_table_paths = find_q_tables(q_table_dir)

    # Check if any Q-tables were found
    if len(q_table_paths) == 0:
        raise FileNotFoundError(f"No Energy-No-Penalty drug Q-tables found in: {q_table_dir}")

    # Evaluate each Q-table in parallel
    max_workers = min(len(q_table_paths), os.cpu_count() or 1)
    logging.info(f"Starting {len(q_table_paths)} evaluations with {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_q_table = {
            executor.submit(evaluate_q_table, q_table_path): q_table_path
            for q_table_path in q_table_paths
        }

        for future in as_completed(future_to_q_table):
            q_table_path = future_to_q_table[future]
            try:
                csv_path = future.result()
                logging.info(f"Finished evaluation for: {os.path.basename(q_table_path)}")
                logging.info(f"Saved CSV: {csv_path}")
            except Exception:
                logging.exception(f"Evaluation failed for: {os.path.basename(q_table_path)}")
