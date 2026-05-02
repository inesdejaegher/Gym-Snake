# Import needed modules
import gym
import gym_snake
import numpy as np
np.bool8 = np.bool_
import random
import time
import datetime
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from helper_func import get_discrete_state,logbook_simulation

# Define different conditions for the simulation loop
conditions = [
    {
        "name": "drug_reward_5_no_growth",
        "n_foods": 1,
        "n_drugs": 1,
        "drug_reward": 5,
        "drug_growth": 0,
    },
    {
        "name": "drug_reward_25_no_growth",
        "n_foods": 1,
        "n_drugs": 1,
        "drug_reward": 25,
        "drug_growth": 0,
    },
    {
        "name": "drug_reward_100_no_growth",
        "n_foods": 1,
        "n_drugs": 1,
        "drug_reward": 100,
        "drug_growth": 0,
    },
]

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')



def run_simulation(condition):

    # Initialise storage of simulation results
    csv_name = f"{condition['name']}_TIME_{datetime.datetime.now().strftime('%d_%m_%Y_%H-%M-%S')}.csv"

    # Dynamically locate the Results folder one directory up from this script
    csv_dir = os.path.join(os.path.dirname(__file__), "..", "Results", "Drugs_No_Growth")
    os.makedirs(csv_dir, exist_ok=True)
    full_csv_path = os.path.join(csv_dir, csv_name)

    # Initialize the drug simulation environment
    env = gym.make('snake-v0')
    
    # Access the unwrapped base environment to safely change parameters before reset
    base_env = env.unwrapped
    base_env.grid_size = [10, 10]  # Smaller grid speeds up initial tabular learning
    base_env.n_foods = condition["n_foods"]
    base_env.n_drugs = condition["n_drugs"]
    base_env.drug_reward = condition["drug_reward"]
    base_env.drug_growth = condition["drug_growth"]

    
    # ----- Q-Learning Hyperparameters -----
    alpha = 0.1             # Learning rate: How quickly the agent abandons old beliefs for new ones
    gamma = 0.95            # Discount factor: How much the agent cares about long-term vs short-term rewards (0 to 1)
    epsilon = 1.0           # Exploration rate: Starts at 100% so the agent completely randomizes its first games
    epsilon_min = 0.01      # The minimum randomness we allow, ensuring it always explores a tiny bit
    epsilon_decay = 0.995   # Epsilon decays this much every episode, slowly transitioning from exploration to exploitation
    episodes = 1000         # Total games to play
    
    # Initialize the Q-table
    # A dictionary where:
    # - Key = The discrete state tuple we extracted above
    # - Value = A numpy array of 4 numbers, representing the "expected value" (Q-value) of moving UP, RIGHT, DOWN, LEFT
    q_table = {}

    # Loop over the number of episodes (games)
    for episode in range(episodes):
        
        # Reset the environment at the start of each game
        env.reset()
        
        # Get the initial discrete state
        state = get_discrete_state(env)
        
        done = False
        total_reward = 0
        drugs_eaten_this_ep = 0
        food_eaten_this_ep = 0
        snake_length = 0
        
        # Play the game until the snake dies (done becomes True)
        while not done:
            
            # If we encounter a state we've never seen, add it to our dictionary with 0 expected value for all 4 actions
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)
            
            # ACTION SELECTION (Epsilon-Greedy)
            # Generate a random float between 0 and 1. If it's less than epsilon, we EXPLORE.
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Pick a random direction
            else:
                # Otherwise, we EXPLOIT. We look at our Q-table and pick the direction with the highest expected value.
                action = int(np.argmax(q_table[state]))
                
            # Take the action in the environment and see what happens
            obs, reward, done, info = env.step(action)
            
            # Track consumed drugs by looking at the info dictionary returned by the environment
            if info.get("drug_eaten", False):
                drugs_eaten_this_ep += 1
                
            # Track consumed food by checking the reward. (Subtract drug reward to isolate food reward)
            if reward - (base_env.drug_reward if info.get("drug_eaten", False) else 0) > 0:
                food_eaten_this_ep += 1
            
            # Keep track of the snake's length.
            # We only update it if it's > 0 because on the final frame when the snake dies, 
            # the environment deletes the snake object entirely and returns a length of 0.
            if info.get("snake_length", 0) > 0:
                snake_length = info.get("snake_length", 0)

            # Read the new state of the board after moving
            next_state = get_discrete_state(env)
            
            # Make sure this new state is also tracked in our table
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)
                
            # Determine the highest possible Q-value we can get from the NEXT state.
            # If the game is over, the future value is 0.
            max_next_q = 0 if done else np.max(q_table[next_state])
            
            # THE BELLMAN EQUATION - This is where the actual "learning" happens.
            # We update the value of the action we just took using the reward we got, plus our expected future rewards.
            # Formula: New_Q = Old_Q + LearningRate * (ImmediateReward + (Discount * MaxFutureQ) - Old_Q)
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max_next_q - q_table[state][action])
            
            # Progress the state tracker
            state = next_state
            total_reward += reward
            
            # Visually render the environment for the last 5 episodes so we can watch what it learned
            if episode >= episodes - 5:
                env.render()
                time.sleep(0.05) # Slow it down slightly so we can watch it
                
        # --- SAVE EPISODE RESULTS ---
        logbook_simulation(full_csv_path, episode, drugs_eaten_this_ep, food_eaten_this_ep, total_reward, epsilon, snake_length)

        # At the end of every episode, decay epsilon so the agent explores less as it gets smarter
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress to the console every 100 episodes
        if (episode + 1) % 100 == 0:
            logging.info(f"Episode {episode + 1}/{episodes} | Epsilon: {epsilon:.3f} | Total Known States: {len(q_table)}")
            
    logging.info(f"Training complete for: {condition['name']}")
    
    # Clean up the rendering window
    env.close()

if __name__ == "__main__":
    for condition in conditions:
        run_simulation(condition)