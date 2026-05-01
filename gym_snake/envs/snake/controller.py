from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import numpy as np

class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.

    snake.py --> managing of data structure of a single snake
    grid.py --> managing of the pixel representation of the board
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        """
        Initialisation function
        --> When a new Controller object is instantiated, this function sets up the entire game board

        """

        # Validation --> ensure that the game is mathematically possible
        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes    # Counter for the amount of snakes we have
        self.grid = Grid(grid_size, unit_size, unit_gap)    # Initialise the grid, passing along the dimensions and the unit sizes

        # Initialise the snakes
        self.snakes = []
        self.dead_snakes = []
        # Loop through the amount of snakes we want
        for i in range(1,n_snakes+1):
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]   # Initialise the start coordinates of the snakes
            self.snakes.append(Snake(start_coord, snake_size))  # Initialise the snakes themselves and append them to the list of snakes we have
            color = [self.grid.HEAD_COLOR[0], 255, i*10]    # If we work with multiple snakes --> give each head a different color  
            self.snakes[-1].head_color = color  # Assign the colors to the different snakes
            self.grid.draw_snake(self.snakes[-1], color)    # Paint the snakes on the board
            self.dead_snakes.append(None)   # Set up the death tracking of the snake

        # Initial food spawn when the game is starting
        if not random_init: # random_init = False --> food is placed in fixed, highly predictable locations
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5] # Calculate specific starting coordinate for food
                self.grid.place_food(start_coord)   # Call function to forcibly draw food at the calculated coordinate
        else:
            for i in range(n_foods):
                self.grid.new_food()    # Draw food at random coordinates at the start of the game

    def move_snake(self, direction, snake_idx):
        """
        Function responsible for the mechanical execution of the first half of the snake's movement.
        It updates the visual grid and the snake's internal state to simulate moving forward, but it 
        deliberately does not check the consequences of that move (like dying or eating food). 
        Those checks are deferred to the move_result function.
        """
        
        # Retrieve the Snake object that we are working with
        # If the snake has previously died --> slot in the list is None --> we return nothing
        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Function is called immediately after move_snake() to determine what happened as a result of the move executed.
        
        """

        # Retrieve the Snake object that we are working with
        # If the snake has previously died --> slot in the list is None --> we return 0
        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake --> has snake's head landed on a wall or another part of the snake?
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]    # move snake to the dead list
            self.snakes[snake_idx] = None   # replace snake object in active snakes list with a None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1 # Reward to penalise the agent for the death

        # Check for reward --> so if snake didn't die, did it take a reward?
        elif self.grid.food_space(snake.head):
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR) # Redraws the visual connection between the newly restored tail and the next body part.
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space & paints head color over the food tile
            reward = 1
            self.grid.new_food() # since old food has been eating --> spawn new food on the grid

        # If the snake did not die and did not take food, this happens  
        else:
            reward = 0

            # If the snake has taken a drug --> it needs to grow for multiple steps, which is taken care of here
            if snake.growth_pending > 0:
                snake.growth_pending -= 1
                self.grid.draw(snake.body[0], self.grid.BODY_COLOR)
                self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            # If it doesn't need to grow, the last part of the tail is popped out of the deque
            else:
                empty_coord = snake.body.popleft()
                self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)

            self.grid.draw(snake.head, snake.head_color)


        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        # The final calculated reward (1, 0, or -1) is returned.
        return reward

    def kill_snake(self, snake_idx):
        """
        Cleaning up death snake from game and subtracts from the snake_count 
        Dead snake body stays on the board until this function is called
        """
        # Sanity check that we are for sure cleaning up a dead snake
        assert self.dead_snakes[snake_idx] is not None

        # Pass head coordinate of the dead snake to the erase function
        self.grid.erase(self.dead_snakes[snake_idx].head)
        # Pass body of dead snake to the erase_snake_body function
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])

        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1

    def step(self, directions):
        """
        Function acts as the primary interface between the RL environment and the game logic
        --> it is the "tick" or "frame advance" mechanism of the game.
        Every time the RL agent decides on an action (a direction to move), 
        that action is passed into this function to calculate the new state of the game.
        
        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if type(directions) == type(int()) or len(directions) == 1:
                return self.grid.grid.copy(), 0, True, {"snakes_remaining":self.snakes_remaining}
            else:
                return self.grid.grid.copy(), [0]*len(directions), True, {"snakes_remaining":self.snakes_remaining}

        rewards = []

        # Input normalisation
        if type(directions) == type(int()):
            directions = [directions]

        # Iterate through every snake in the game, processing their intended directions one by one
        for i, direction in enumerate(directions):

            # Check if the snake died in the previous step
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                # If snake died --> wipe dead body of the board
                self.kill_snake(i)
                # The game leaves a dead snake's body on the grid for exactly one frame so the RL agent can visually "see" the crash, and this line is what cleans it up on the subsequent frame.
            
            # Execute the mechanical math to push the snake forward one unit in the requested direction.
            # Erase the visual tail and paints over the old head.
            self.move_snake(direction,i)

            # Evaluate where the snake just landed & calculate the reward for that outcome and append it to the rewards list
            rewards.append(self.move_result(direction, i))

        # Check the end game conditions once again
        done = self.snakes_remaining < 1 or self.grid.open_space < 1
        if len(rewards) == 1:
            # Return the standard 4-part tuple expected by OpenAI Gym
            return self.grid.grid.copy(), rewards[0], done, {"snakes_remaining":self.snakes_remaining}
        else:
            return self.grid.grid.copy(), rewards, done, {"snakes_remaining":self.snakes_remaining}
        
        # self.grid.grid.copy() --> The actual raw pixel data (RGB array) of the game board
        # rewards --> Returns either a single integer (for one snake) or a list of integers (for multiple snakes)
        # done --> A boolean indicating if the episode has ended
        # {"snakes_remaining":self.snakes_remaining} --> A dictionary tracking metadata, specifically how many snakes are still alive
