import gym
import gym_snake
import numpy as np
import random

def get_discrete_state(env):
    """
    This function extracts a simplified, discrete state from the environment.
    Tabular Q-learning cannot handle raw 150x150 pixel arrays because the state space 
    would be infinitely large. Instead, we extract 6 key features:
    1. Is food to the left, right, or same X column? (-1, 0, 1)
    2. Is food up, down, or same Y row? (-1, 0, 1)
    3-6. Is there a wall or tail immediately Up, Right, Down, or Left? (0 or 1)
    """
    
    # Access the underlying game controller to extract coordinates safely
    controller = env.controller
    
    # Get the primary snake
    snake = controller.snakes[0]
    
    # If the snake just died this frame, its object is moved to the dead_snakes list
    if snake is None:
        snake = controller.dead_snakes[0]
        
    # If it is still None (edge case safeguard), return a dummy state
    if snake is None:
        return (0, 0, 1, 1, 1, 1)

    # Grab the current (x,y) grid coordinates of the snake's head
    head_x, head_y = int(snake.head[0]), int(snake.head[1])
    
    # Find the coordinates of the food by scanning the 15x15 logical grid
    food_x, food_y = head_x, head_y
    for x in range(env.grid_size[0]):
        for y in range(env.grid_size[1]):
            # We use the internal grid helper to check if a specific tile contains food
            if controller.grid.food_space((x, y)):
                food_x, food_y = x, y
                break
                
    # 1. Calculate relative X direction of the food (-1 for left, 1 for right, 0 for aligned)
    dir_x = 1 if food_x > head_x else (-1 if food_x < head_x else 0)
    
    # 2. Calculate relative Y direction of the food (-1 for up, 1 for down, 0 for aligned)
    dir_y = 1 if food_y > head_y else (-1 if food_y < head_y else 0)
    
    # 3-6. Check immediate dangers 1 block away using the grid's death checker
    # UP is y-1, RIGHT is x+1, DOWN is y+1, LEFT is x-1
    danger_up = int(controller.grid.check_death((head_x, head_y - 1)))
    danger_right = int(controller.grid.check_death((head_x + 1, head_y)))
    danger_down = int(controller.grid.check_death((head_x, head_y + 1)))
    danger_left = int(controller.grid.check_death((head_x - 1, head_y)))
    
    # Return the simplified state as a tuple so it can be used as a dictionary key in our Q-table
    return (dir_x, dir_y, danger_up, danger_right, danger_down, danger_left)


if __name__ == "__main__":
    # Initialize the baseline environment
    env = gym.make('snake-v0')
    
    # Enforce the baseline rules exactly as requested: 1 food, 0 drugs
    env.n_foods = 1
    env.n_drugs = 0
    
    # Q-Learning Hyperparameters
    alpha = 0.1             # Learning rate: How much new information overrides old information
    gamma = 0.9             # Discount factor: How much the agent cares about future long-term rewards vs immediate rewards
    epsilon = 1.0           # Exploration rate: Starts at 100% (completely random actions to explore the board)
    epsilon_min = 0.01      # The minimum exploration rate so the agent occasionally tries new things even when trained
    epsilon_decay = 0.995   # The rate at which we decay epsilon every episode (gradually shifting from explore to exploit)
    episodes = 2000         # Total number of games the agent will play to learn
    
    # Initialize the Q-table as an empty dictionary. 
    # Keys will be state tuples, values will be numpy arrays of length 4 (representing Q-values for UP, RIGHT, DOWN, LEFT)
    Q_table = {}

    print("Starting Q-Learning Training...")

    # Main training loop
    for episode in range(episodes):
        
        # Reset the environment at the start of each game and get the initial raw pixel observation
        obs = env.reset()
        
        # Convert the complex pixel observation into our simple discrete state
        state = get_discrete_state(env)
        
        # Track the total score for this specific episode for logging
        total_reward = 0
        done = False
        
        # Play the game until the snake dies (done = True)
        while not done:
            
            # If we've never seen this state before, initialize its action values to [0, 0, 0, 0]
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)
            
            # Action Selection: Epsilon-Greedy Strategy
            # Generate a random number. If it's less than epsilon, we EXPLORE (take a random action)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # Otherwise, we EXPLOIT (take the action with the highest Q-value for our current state)
                action = np.argmax(Q_table[state])
                
            # Pass the chosen action to the environment to advance the game by 1 frame
            next_obs, reward, done, info = env.step(action)
            
            # Get the new discrete state resulting from that action
            next_state = get_discrete_state(env)
            
            # Ensure the next state exists in our Q-table
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n)
                
            # Calculate the maximum possible Q-value we can get from the next state.
            # If the game is 'done' (snake died), the future value is 0 because the game is over.
            max_next_q = 0 if done else np.max(Q_table[next_state])
            
            # --- THE BELLMAN EQUATION ---
            # Update the Q-value for our previous state-action pair.
            # Formula: Old_Q + LearningRate * (ImmediateReward + DiscountFactor * MaxFutureQ - Old_Q)
            Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * max_next_q - Q_table[state][action])
            
            # Advance the state tracker for the next loop iteration
            state = next_state
            
            # Add the reward to our episode tracker
            total_reward += reward
            
            # If we are in the last 5 episodes, visually render the game so we can watch the trained agent!
            if episode >= episodes - 5:
                env.render(frame_speed=0.05)
                
        # At the end of every episode, decay epsilon so the agent explores less and exploits more
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Log progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} | Epsilon: {epsilon:.3f} | Unique States Discovered: {len(Q_table)}")
            
    print("Training Finished!")
