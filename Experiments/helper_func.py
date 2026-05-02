import gym
import gym_snake
import numpy as np
import random
import time

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
    
    # 4. Check for immediate dangers (walls or own body) using the grid's death checker
    danger_up = int(controller.grid.check_death((head_x, head_y - 1)))
    danger_right = int(controller.grid.check_death((head_x + 1, head_y)))
    danger_down = int(controller.grid.check_death((head_x, head_y + 1)))
    danger_left = int(controller.grid.check_death((head_x - 1, head_y)))
    
    # Return the simple tuple which will be used as a dictionary key in our Q-Table
    return (dir_x, dir_y, danger_up, danger_right, danger_down, danger_left)