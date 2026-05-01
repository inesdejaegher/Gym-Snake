import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: see matplotlib documentation for installation "
        "https://matplotlib.org/faq/installing_faq.html#installation".format(e)
    )


class SnakeEnv(gym.Env):
    """
    Official OpenAI Gym environment wrapper for the underlying Snake game.
    This class translates the game mechanics (from controller.py and grid.py) into the standard format 
    expected by RL algorithms (specifically the 'reset', 'step', and 'render' methods)

    The class also introduces a completely custom game mechanic "Drugs". " It overlays this new mechanic 
    on top of the base game without actually modifying the underlying Grid or Controller classes.
    """
    metadata = {'render.modes': ['human']}

    DRUG_COLOR = np.array([255, 0, 0], dtype=np.uint8)   # red
    DRUG_REWARD = 6
    DRUG_GROWTH = 6
    N_DRUGS = 0

    def __init__(
        self,
        grid_size=[15, 15],
        unit_size=10,
        unit_gap=1,
        snake_size=3,
        n_snakes=1,
        n_foods=1,
        n_drugs=None,
        random_init=True,
        drug_reward=None,
        drug_growth=None):
        """
        Initialisation function
        --> Setting Instance Variables = assigns grid dimensions, game settings and the drug parameters to the instance
        --> Fallback Logic = For drug_reward and drug_growth, it checks if you provided a custom value during initialization. 
            If you left them as None, it falls back to the default class constants (self.DRUG_REWARD and self.DRUG_GROWTH).
        --> Gym Spaces: It defines the action_space (4 discrete directional moves) and the observation_space (the mathematical 
            shape and color boundaries of the visual game grid) which are required by RL algorithms to understand the environment.
        --> Underlying Game Controller: It initializes the core game Controller (which handles the basic snake and food math).
        """
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.n_drugs = self.N_DRUGS if n_drugs is None else n_drugs
        self.drug_reward = self.DRUG_REWARD if drug_reward is None else drug_reward
        self.drug_growth = self.DRUG_GROWTH if drug_growth is None else drug_growth
        self.viewer = None
        self.random_init = random_init

        self.action_space = spaces.Discrete(4)

        # IMPORTANT: Controller does NOT take n_drugs
        controller = Controller(
            self.grid_size,
            self.unit_size,
            self.unit_gap,
            self.snake_size,
            self.n_snakes,
            self.n_foods,
            random_init=self.random_init
        )

        grid = controller.grid
        self.observation_space = spaces.Box(
            low=np.min(grid.COLORS),
            high=np.max(grid.COLORS),
            shape=grid.grid.shape,
            dtype=np.uint8
        )

        self.controller = None
        self.last_obs = None
        self.drug_positions = []

    def _get_occupied_positions(self):
        """
        Function that scans the current game board and returns a collection (a set in this case) of all the (x,y) coordinates
        that are currently "taken" by an entity in the game.
        """
        # Create empty set --> set because checking if an item exists in a set is extremely fast + it prevents duplicates.
        occupied = set()

        # --- POSITIONS OF THE SNAKES ---
        # Loop through every snake managed by the game's controller
        for snake in self.controller.snakes:
            # Skip dead snakes
            if snake is None:
                continue

            # If snake is alive --> grab (x,y) coordinates of the head and add to the set
            occupied.add((int(snake.head[0]), int(snake.head[1])))
            # Loop over body parts
            for body_part in snake.body:
                # Grab the (x,y) coordinates of each body part and add to the set
                occupied.add((int(body_part[0]), int(body_part[1])))

        # --- POSITIONS OF THE FOODS
        grid = self.controller.grid
        # Iterate over every single tile in the grid
        for x in range(int(grid.grid_size[0])):
            for y in range(int(grid.grid_size[1])):
                # Check if the tile contains food
                if grid.food_space((x, y)):
                    # If it does --> append to set
                    occupied.add((x, y))

        return occupied

    def _spawn_drugs(self):
        """
        Function is responsible for figuring out exactly where on the grid the drugs should be placed when the 
        environment starts or resets. It ensures that drugs are placed randomly but safely (not overlapping with 
        the snake, food, or other drugs).
        """
        # Reset drug positions (empty list)
        self.drug_positions = []

        # Find the spots that are occupied
        occupied = self._get_occupied_positions()

        # Define the grid size
        grid_w, grid_h = self.grid_size

        # While loop for as long as we have less drugs than the specified amount
        while len(self.drug_positions) < self.n_drugs:
            # Generate random coordinates
            x = np.random.randint(0, grid_w)
            y = np.random.randint(0, grid_h)

            # Check if the random coordinates are occupied
            if (x, y) not in occupied:
                # If they are not occupied, add the coordinates to the drug positions list
                self.drug_positions.append(np.array([x, y]))
                # Add the coordinates to the occupied set
                occupied.add((x, y))

    def _draw_drugs(self, frame):
        """
        Function responsible for drawing/visualising the drugs on the game board.
        """
        # Loop through every drug position
        for drug_pos in self.drug_positions:
            # Get the coordinates of the drug positions (in pixels)
            x = int(drug_pos[0] * self.unit_size)
            y = int(drug_pos[1] * self.unit_size)

            end_x = x + self.unit_size - self.unit_gap
            end_y = y + self.unit_size - self.unit_gap

            # Draw the drug on the frame
            frame[y:end_y, x:end_x, :] = self.DRUG_COLOR

        return frame

    def _check_drug_collision(self):
        """
        Function that acts as the event listener and handler for when a snake interacts with a drug. 
        It is called every single time the game advances by one frame (inside the step() method)
        """
        # Grab the primary snake
        snake = self.controller.snakes[0]
        # Check if the snake is currently dead
        if snake is None:
            # If it is dead, return 0 (no reward) and False (no drug eaten)
            return 0, False

        # Iterate over all drug positions
        for i, drug_pos in enumerate(self.drug_positions):
            # Check if the head of the snake and the drug coordinates are the same
            if np.array_equal(snake.head, drug_pos):
                # Growth effect of eating the drug --> pending growth increases
                snake.growth_pending += self.drug_growth

                # Remove the drug that was eaten
                self.drug_positions.pop(i)

                # Respawn one new drug somewhere else
                occupied = self._get_occupied_positions()
                occupied.update((int(p[0]), int(p[1])) for p in self.drug_positions)

                grid_w, grid_h = self.grid_size
                while True:
                    x = np.random.randint(0, grid_w)
                    y = np.random.randint(0, grid_h)
                    if (x, y) not in occupied:
                        self.drug_positions.append(np.array([x, y]))
                        break

                # Return the reward from eating the drug and True (indicating that a drug was eaten)
                return self.drug_reward, True

        # If no drug was eaten, return 0 reward and False
        return 0, False

    def step(self, action):
        """
        Core of the Gym environment interaction loop
        --> is called every step by the RL agent to advance the game
        """
        # Call the action of the agent and get the outputs
        # --> moves snake
        # --> checks for standard food consumption or death
        # --> returns base result
        self.last_obs, reward, done, info = self.controller.step(action)

        # Add drug reward on top of normal controller reward
        drug_reward, drug_eaten = self._check_drug_collision()
        
        # Combine rewards
        reward += drug_reward

        # Update information
        info["drug_eaten"] = drug_eaten
        info["snake_length"] = self.get_snake_length()

        # Return the complete, updated state to the agent (following the OpenAI Gym format)
        return self.last_obs, reward, done, info

    def get_snake_length(self):
        """
        Helper utility funtion that calculates and returns the current length of the snake
        """

        # Find the current snake
        snake = self.controller.snakes[0]
        if snake is None:
            # If snake has died --> game keeps snake data for 1 extra frame in a dead_snake list
            # retrieve this information to calculate last length
            snake = self.controller.dead_snakes[0]

        # No snake found --> return 0
        if snake is None:
            return 0
        
        return 1 + len(snake.body)

    def reset(self):
        """
        Function is called to start a brand new game (or "episode"). 
        When an RL agent dies or wins, it calls this method to reset the environment back to its initial state.
        """

        # Reinitialise the controller
        self.controller = Controller(
            self.grid_size,
            self.unit_size,
            self.unit_gap,
            self.snake_size,
            self.n_snakes,
            self.n_foods,
            random_init=self.random_init
        )

        # Save the view --> grab a copy of the pixel data from the newly created grid and saves it as self.last_obs (the first observation)
        self.last_obs = self.controller.grid.grid.copy()
        # Respawn the drugs
        self._spawn_drugs()
        # Return the last observation such that the RL agent can see the board
        return self.last_obs

    def render(self, mode='human', close=False, frame_speed=.1):
        """
        Function is responsible for visually displaying the game on your screen so you can watch the agent play. 
        It uses the matplotlib library to draw the frames.
        """
        
        # If it's the first time that the render is called, it initialises a matplotlib figure and viewer
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()

        self.viewer.clear()

        # Draw the grid
        frame = self.last_obs.copy()
        # Draw the drugs
        frame = self._draw_drugs(frame)

        # Display the frame
        self.viewer.imshow(frame)
        plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass
