import sys
from gym_maze.envs.maze_env import MazeEnv

# Initialize the "maze" environment
env = MazeEnv()

## Defining the simulation related constants
NUM_EPISODES = 1
MAX_T = 5000
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = False

for episode in range(NUM_EPISODES):
    # Reset the environment
    obv = env.reset()

    for t in range(MAX_T):

        # choose a random action
        action = env.action_space.sample()

        # execute the action
        obv, reward, done, _ = env.step(action)

        # render
        simulation_stopped = env.render()

        if done:
            break

        if simulation_stopped:
            sys.exit()
