import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
from gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human"],
        "video.frames_per_second": 60
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self):

        self.maze_view = MazeView2D("maze2d.npy")
        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size))
        high =  np.array(self.maze_size) - np.ones(len(self.maze_size))
        self.observation_space = spaces.Box(low, high)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self._seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.state = self.maze_view.move_robot

        info = {}

        return self.state, reward, done, info

    def _reset(self):
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update()
