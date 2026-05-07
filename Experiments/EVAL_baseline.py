import pickle
import time
import logging
import warnings
import os
import datetime
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
# Path to stored Q-table
q_table_name = "q_table_baseline_EP_5000_TIME_04_05_2026_15-34-11.pkl"
csv_name = f"Evaluation_Results_logbook_" + q_table_name.replace(".pkl", ".csv")

# Number of episodes we want to do evaluation on
eval_episodes = 100
max_steps_without_consumption = 100


# ----- STORAGE FOLDER FOR RESULTS -----
q_table_dir = os.path.join(os.path.dirname(__file__), "..", "Q-Tables", "Baseline")
os.makedirs(q_table_dir, exist_ok=True) 
q_table_path = os.path.join(q_table_dir, q_table_name) 

csv_dir = os.path.join(os.path.dirname(__file__), "..", "Results", "Baseline")
os.makedirs(csv_dir, exist_ok=True) # Create the folder if it doesn't exist
full_csv_path = os.path.join(csv_dir, csv_name) # Combine folder and file name

# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the trained Q-table
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    logging.info(f"Loaded Q-table with {len(q_table)} known states.")

    # ------------------------------------------------------------------
    # ----- Initialize environment (MUST MATCH TRAINING SETTINGS!) -----
    # ------------------------------------------------------------------
    env = gym.make('snake-v0')
    base_env = env.unwrapped
    base_env.grid_size = [10, 10]  # Smaller grid speeds up initial tabular learning
    base_env.n_foods = 1           # Baseline scenario: strictly 1 food
    base_env.n_drugs = 0           # Baseline scenario: no drugs
    base_env.drug_reward = 0       # Baseline scenario: drugs have no effect
    base_env.drug_growth = 0       # Baseline scenario: drugs give no growth

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
        steps_without_food = 0
        loop = 0
        
        while not done:
            # --- PURE EXPLOITATION ---
            # If the agent encounters a state it recognizes, take the best action
            if state in q_table:
                action = int(np.argmax(q_table[state]))
            else:
                # If it encounters a completely new state during evaluation, 
                # we just pick a random action as a fallback.
                action = env.action_space.sample()
                
            # Take the action
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
                steps_without_food = 0
            else:
                steps_without_food += 1

            # ----- TRACKING SNAKE LENGTH ----
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
            # Prevent infinite loops if the agent gets stuck going in circles
            if steps_without_food > max_steps_without_consumption:
                logging.info("Agent stuck in a loop. Forcing episode end.")
                loop = 1
                break

        # --- SAVE EPISODE RESULTS ---
        logbook_simulation(full_csv_path, episode, drugs_eaten_this_ep, food_eaten_this_ep, total_reward, snake_length, steps, loop)
                
        logging.info(
            f"Baseline | Evaluation Episode {episode + 1}/{eval_episodes} finished "
            f"| Total Reward: {total_reward} | Drugs: {drugs_eaten_this_ep} "
            f"| Food: {food_eaten_this_ep} | Snake Length: {snake_length} | Steps: {steps}"
        )

    env.close()
    logging.info("Evaluation Complete!")