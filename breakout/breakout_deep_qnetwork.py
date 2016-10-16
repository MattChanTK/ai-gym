import gym
from tf_neural_net import PixelsCNN
import tensorflow as tf
import sys
import numpy as np
import math
import random


# Initialize the Breakout game environment
env = gym.make("Breakout-v0")
env.render()

NUM_EPISODES = 100
MAX_T = 200

print(env.get_action_meanings())
print(env.action_space)
print(env.observation_space)

# Actions are down, stay still, up
ACTION_COUNT = env.action_space.n
STATE_FRAMES = 4
SCREEN_X, SCREEN_Y = 80, 80

# Instantiate the neural network
neural_net = PixelsCNN(image_size=(SCREEN_X, SCREEN_Y), num_outputs=ACTION_COUNT, num_frames=STATE_FRAMES)

for episode in range(NUM_EPISODES):

    # Reset the environment
    obv = env.reset()

    for t in range(MAX_T):

        env.render()

        # Execute the action
        obv, reward, done, _ = env.step(2)
