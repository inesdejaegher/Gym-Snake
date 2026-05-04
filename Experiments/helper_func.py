import gym
import gym_snake
import numpy as np
np.bool8 = np.bool_
import random
import time
import os
import csv
import matplotlib.pyplot as plt

def get_discrete_state(env):
    """
    Q-learning requires discrete (limited and countable) states. 
    If we used the raw pixel grid, the number of possible states would be near infinite.
    Instead, we extract 6 simple features to represent the state of our snake:
    1. Is the food left, right, or straight on the X axis? (-1, 0, 1)
    2. Is the food up, down, or straight on the Y axis? (-1, 0, 1)
    3-6. Are the immediate tiles (Up, Right, Down, Left) deadly walls/tails? (0 or 1)
    """
    
    # Access the game logic controller inside the gym wrapper
    controller = env.unwrapped.controller
    
    # Get the primary snake's object
    snake = controller.snakes[0]
    
    # If the snake just died, it gets moved to a dead_snakes list for 1 frame.
    if snake is None:
        snake = controller.dead_snakes[0]
        
    # Fallback if somehow still missing (edge case)
    if snake is None:
        return (0, 0, 1, 1, 1, 1)

    # 1. Grab head coordinates
    head_x, head_y = int(snake.head[0]), int(snake.head[1])
    
    # 2. Find food coordinates by scanning the grid
    food_x, food_y = head_x, head_y
    for x in range(int(controller.grid.grid_size[0])):
        for y in range(int(controller.grid.grid_size[1])):
            # Check if this specific tile contains food
            if controller.grid.food_space((x, y)):
                food_x, food_y = x, y
                break
                
    # 3. Calculate relative X & Y distances to the food
    dir_x = 1 if food_x > head_x else (-1 if food_x < head_x else 0)
    dir_y = 1 if food_y > head_y else (-1 if food_y < head_y else 0)
    
    #4. Calculate relative X & Y distances to the drug
    drug_dir_x, drug_dir_y = 0, 0
    # Access the list of drug coordinates from your custom environment wrapper
    drug_positions = env.unwrapped.drug_positions 
    
    if len(drug_positions) > 0:
        # Grab the X and Y of the first drug on the board
        drug_x = int(drug_positions[0][0])
        drug_y = int(drug_positions[0][1])
        
        # Calculate relative X & Y distances to the drug
        drug_dir_x = 1 if drug_x > head_x else (-1 if drug_x < head_x else 0)
        drug_dir_y = 1 if drug_y > head_y else (-1 if drug_y < head_y else 0)
        
    # 4. Check for immediate dangers (walls or own body) using the grid's death checker
    danger_up = int(controller.grid.check_death((head_x, head_y - 1)))
    danger_right = int(controller.grid.check_death((head_x + 1, head_y)))
    danger_down = int(controller.grid.check_death((head_x, head_y + 1)))
    danger_left = int(controller.grid.check_death((head_x - 1, head_y)))
    
    # Return the simple tuple which will be used as a dictionary key in our Q-Table
    return (dir_x, dir_y, danger_up, danger_right, danger_down, danger_left)

def logbook_simulation(file_path, episode, n_drugs_consumed, n_food_consumed, total_reward,epsilon, snake_length):
    """
    Makes a logbook of the different parameters during a single episode
    and appends the result to a CSV file.
    """

    # ----- PREFERENCE RATIO -----
    # Calculate ratio safely
    if n_food_consumed == 0:
        if n_drugs_consumed == 0:
            ratio = 0.0
        else:
            ratio = float('inf') # Agent only ate drugs, no food
    else:
        ratio = n_drugs_consumed / n_food_consumed


    # ----- SAVE EPISODE RESULTS -----
    # Check if the file exists so we know if we need to write headers
    file_exists = os.path.isfile(file_path)

    # Open the CSV file in append mode ('a')
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers if it's a brand new file
        if not file_exists:
            writer.writerow(['Episode', 'Drugs_Consumed', 'Food_Consumed', 'Preference_Ratio', 'Total_Reward', 'Epsilon', 'Snake_Length'])
            
        # Write the data for the current episode
        writer.writerow([episode, n_drugs_consumed, n_food_consumed, ratio, total_reward, epsilon, snake_length])
        
    return ratio, total_reward, epsilon, snake_length

def plot_preference_ratio_from_csv(file_path):
    """
    Reads the generated CSV file and plots the Preference Ratio over episodes.
    """
    if not os.path.isfile(file_path):
        print(f"Error: Could not find {file_path}")
        return

    episodes = []
    ratios = []

    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            episodes.append(int(row['Episode']))
            ratios.append(float(row['Preference_Ratio']))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, ratios, label='Drug/Food Ratio', color='purple', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Preference Ratio (Drugs / Food)')
    plt.title('Agent Preference Ratio over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reward_from_csv(file_path, window_size=20):
    """
    Reads the generated CSV file and plots the Total Reward over episodes,
    including a moving average to smooth out the noise.
    """
    if not os.path.isfile(file_path):
        print(f"Error: Could not find {file_path}")
        return

    episodes = []
    rewards = []

    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            episodes.append(int(row['Episode']))
            rewards.append(float(row['Total_Reward']))

    plt.figure(figsize=(10, 5))
    # Plot the raw, noisy rewards slightly transparent
    plt.plot(episodes, rewards, label='Total Reward (Raw)', color='green', alpha=0.3)
    
    # Calculate and plot the moving average if we have enough data
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(episodes[window_size-1:], moving_avg, label=f'{window_size}-Episode Moving Avg', color='darkgreen', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Total Reward over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epsilon_from_csv(file_path, window_size=20):
    """
    Reads the generated CSV file and plots the Epsilon decay over episodes.
    """
    if not os.path.isfile(file_path):
        print(f"Error: Could not find {file_path}")
        return

    episodes = []
    epsilons = []

    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            episodes.append(int(row['Episode']))
            epsilons.append(float(row['Epsilon']))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, epsilons, label='Epsilon (Exploration Rate)', color='darkblue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Agent Exploration Rate (Epsilon) Decay over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_snake_length_from_csv(file_path, window_size=20):
    """
    Reads the generated CSV file and plots the Snake Length over episodes.
    """

    if not os.path.isfile(file_path):
        print(f"Error: Could not find {file_path}")
        return

    episodes = []
    snake_lengths = []

    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            episodes.append(int(row['Episode']))
            snake_lengths.append(float(row['Snake_Length']))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, snake_lengths, label='Snake Length', color='blue', linewidth=2, alpha=0.3)

    # Calculate and plot the moving average if we have enough data
    if len(snake_lengths) >= window_size:
        moving_avg = np.convolve(snake_lengths, np.ones(window_size)/window_size, mode='valid')
        plt.plot(episodes[window_size-1:], moving_avg, label=f'{window_size}-Episode Moving Avg', color='darkblue', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Snake Length at End of Episode')
    plt.title('Snake Length at End of Episode over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metric_subplots_from_csv(file_paths, subplot_titles, metric, y_label=None, main_title=None, window_size=20, ncols=None):
    """
    Plots the same metric from multiple CSV files as subplots.

    file_paths: list of CSV paths
    subplot_titles: list of titles, one for each subplot
    metric: CSV column to plot, e.g. 'Total_Reward' or 'Preference_Ratio'
    """
    if len(file_paths) == 0:
        print("Error: No CSV files were provided.")
        return None, None

    if len(file_paths) != len(subplot_titles):
        print("Error: file_paths and subplot_titles must have the same length.")
        return None, None

    if ncols is None:
        ncols = len(file_paths)

    nrows = int(np.ceil(len(file_paths) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, file_path, subplot_title in zip(axes, file_paths, subplot_titles):
        if not os.path.isfile(file_path):
            ax.set_title(subplot_title)
            ax.text(0.5, 0.5, "CSV file not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        episodes = []
        values = []

        with open(file_path, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            if metric not in reader.fieldnames:
                ax.set_title(subplot_title)
                ax.text(0.5, 0.5, f"Column not found: {metric}", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            for row in reader:
                episodes.append(int(row['Episode']))
                values.append(float(row[metric]))

        ax.plot(episodes, values, label=metric, alpha=0.3)

        if len(values) >= window_size:
            moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
            ax.plot(episodes[window_size - 1:], moving_avg, label=f'{window_size}-Episode Moving Avg', linewidth=2)

        ax.set_title(subplot_title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(y_label if y_label is not None else metric)
        ax.legend()
        ax.grid(True)

    for ax in axes[len(file_paths):]:
        ax.set_axis_off()

    if main_title is not None:
        fig.suptitle(main_title)

    plt.tight_layout()
    plt.show()

    return fig, axes[:len(file_paths)]
