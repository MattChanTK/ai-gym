import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    ACTION = {
        "UP":       0,
        "DOWN":     1,
        "LEFT":     2,
        "RIGHT":    3,
    }

    def __init__(self, maze_size=(10,10)):

        # specifying the size of the maze
        assert(isinstance(maze_size, (tuple, list)))
        assert(len(maze_size) == 2) # only support 2D maze right now
        self.maze_size = np.array(maze_size)

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size)) # 2D: (up, down, left, right)
        self.observation_space = spaces.Box(np.zeros(len(self.maze_size)), self.maze_size)

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        pass
    def _reset(self):
        pass
    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            pass

