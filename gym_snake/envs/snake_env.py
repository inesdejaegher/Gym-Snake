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
    metadata = {'render.modes': ['human']}

    DRUG_COLOR = np.array([255, 0, 255], dtype=np.uint8)   # purple
    DRUG_REWARD = 6

    def __init__(
        self,
        grid_size=[15, 15],
        unit_size=10,
        unit_gap=1,
        snake_size=3,
        n_snakes=1,
        n_foods=1,
        n_drugs=0,
        random_init=True
    ):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.n_foods = n_foods
        self.n_drugs = n_drugs
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
        occupied = set()

        # snake head + body
        for snake in self.controller.snakes:
            if snake is None:
                continue

            occupied.add((int(snake.head[0]), int(snake.head[1])))
            for body_part in snake.body:
                occupied.add((int(body_part[0]), int(body_part[1])))

        # built-in food positions
        grid = self.controller.grid
        for x in range(int(grid.grid_size[0])):
            for y in range(int(grid.grid_size[1])):
                if grid.food_space((x, y)):
                    occupied.add((x, y))

        return occupied

    def _spawn_drugs(self):
        self.drug_positions = []
        occupied = self._get_occupied_positions()

        grid_w, grid_h = self.grid_size

        while len(self.drug_positions) < self.n_drugs:
            x = np.random.randint(0, grid_w)
            y = np.random.randint(0, grid_h)

            if (x, y) not in occupied:
                self.drug_positions.append(np.array([x, y]))
                occupied.add((x, y))

    def _draw_drugs(self, frame):
        for drug_pos in self.drug_positions:
            x = int(drug_pos[0] * self.unit_size)
            y = int(drug_pos[1] * self.unit_size)

            end_x = x + self.unit_size - self.unit_gap
            end_y = y + self.unit_size - self.unit_gap

            frame[y:end_y, x:end_x, :] = self.DRUG_COLOR

        return frame

    def _check_drug_collision(self):
        snake = self.controller.snakes[0]
        if snake is None:
            return 0

        for i, drug_pos in enumerate(self.drug_positions):
            if np.array_equal(snake.head, drug_pos):
                # remove eaten drug
                self.drug_positions.pop(i)

                # respawn one new drug somewhere else
                occupied = self._get_occupied_positions()
                occupied.update((int(p[0]), int(p[1])) for p in self.drug_positions)

                grid_w, grid_h = self.grid_size
                while True:
                    x = np.random.randint(0, grid_w)
                    y = np.random.randint(0, grid_h)
                    if (x, y) not in occupied:
                        self.drug_positions.append(np.array([x, y]))
                        break

                return self.DRUG_REWARD

        return 0

    def step(self, action):
        self.last_obs, reward, done, info = self.controller.step(action)

        # add drug reward on top of normal controller reward
        reward += self._check_drug_collision()

        return self.last_obs, reward, done, info

    def reset(self):
        self.controller = Controller(
            self.grid_size,
            self.unit_size,
            self.unit_gap,
            self.snake_size,
            self.n_snakes,
            self.n_foods,
            random_init=self.random_init
        )

        self.last_obs = self.controller.grid.grid.copy()
        self._spawn_drugs()
        return self.last_obs

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()

        self.viewer.clear()

        frame = self.last_obs.copy()
        frame = self._draw_drugs(frame)

        self.viewer.imshow(frame)
        plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass