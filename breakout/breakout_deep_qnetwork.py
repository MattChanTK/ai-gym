import gym
from tf_neural_net import PixelsCNN
import tensorflow as tf
import sys
import numpy as np
from collections import deque
from PIL import Image
import math
import random


# Initialize the Breakout game environment
env = gym.make("Breakout-v0")

NUM_EPISODES = 100
MAX_T = 200

# Actions are down, stay still, up
ACTIONS_COUNT = env.action_space.n
SCREEN_H_ORIG, SCREEN_W_ORIG, NUM_COLOUR_CHANNELS = env.observation_space.shape
STATE_FRAMES = 4
SCREEN_W, SCREEN_H = 80, 80

# Learning-related variables
LEARN_RATE = 1e-6
EXPLORE_RATE_0 = 1.0
MIN_EXPLORE_RATE = 0.05
NUM_EXPLORE_STEPS = 200000
SCORE_HISTORY_SIZE = 1000
OBV_SAMPLES_SIZE = 100000
INIT_OBV_STEPS = 10 #50000
MINI_BATCH_SIZE = 5 # 100
TIME_COST = 0.99

# Instantiate the neural network
neural_network = PixelsCNN(image_size=(SCREEN_W, SCREEN_H), num_outputs=ACTIONS_COUNT, num_frames=STATE_FRAMES)
input_layer, output_layer = neural_network.create_network()

action = tf.placeholder(tf.float32, [None, ACTIONS_COUNT])
target = tf.placeholder(tf.float32, [None])

readout_action = tf.reduce_sum(tf.mul(output_layer, action), reduction_indices=1)
cost = tf.reduce_sum(tf.square(output_layer - readout_action))
train_operation = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)

observations = deque()
last_scores = deque()

def pre_process(screen_image):

    # crop the top and bottom
    screen_image = screen_image[35:195]

    # down sample by a factor of 2
    screen_image = screen_image[::2, ::2]

    # convert to grey scale
    grey_image = np.zeros(screen_image.shape[0:2])
    for i in range(len(screen_image)):
        for j in range(len(screen_image[i])):
            grey_image[i][j] = np.mean(screen_image[i][j])

    if grey_image.shape != (SCREEN_W, SCREEN_H):
        raise ValueError("Screen image must be reduced to (%d, %d). " % (SCREEN_W, SCREEN_H))

    return grey_image.astype(np.float)


def action_from_output(output_array):

    action_id = 0
    for output in output_array:
        if output:
            return action_id
        action_id += 1
    raise ValueError("output_array cannot be all 0's.")


def select_action(last_state, explore_rate):
    new_action = np.zeros([ACTIONS_COUNT])

    if random.random() <= explore_rate:
        # choose an action randomly
        action_index = random.randrange(ACTIONS_COUNT)
    else:
        # choose an action given our last state
        readout_t = session.run(output_layer, feed_dict={input_layer: [last_state]})[0]

        action_index = np.argmax(readout_t)

    new_action[action_index] = 1
    return new_action


def train():
    # sample a mini_batch to train on
    mini_batch = random.sample(observations, MINI_BATCH_SIZE)
    # get the batch variables
    batch_variables = list(zip(*mini_batch))
    prev_states, actions, rewards, curr_states, terminal = batch_variables

    agents_expected_reward = []
    # this gives us the agents expected reward for each action we might take
    agents_reward_per_action = session.run(output_layer, feed_dict={input_layer: curr_states})
    for i in range(len(mini_batch)):
        if terminal[i]:
            # this was a terminal frame so there is no future reward...
            agents_expected_reward.append(rewards[i])
        else:
            agents_expected_reward.append(
                rewards[i] + TIME_COST * np.max(agents_reward_per_action[i]))

    # learn that these actions in these states lead to this reward
    session.run(train_operation,
                feed_dict={ input_layer: prev_states,
                            action: actions,
                            target: agents_expected_reward })


def simulation_begin():

    # initialize variables
    last_output = np.zeros(ACTIONS_COUNT)
    last_output[1] = 1
    last_state = None
    explore_rate = EXPLORE_RATE_0
    time = 0

    next_action = env.action_space.sample()
    env.reset()

    for episode in range(NUM_EPISODES):

        # Reset the environment

        env.render()

        # Execute the action
        obv, reward, done, _ = env.step(next_action)

        if done:
            env.reset()

        terminal = True

        if reward != 0.0:
            terminal = True
            last_scores.append(reward)
            if len(last_scores) > SCORE_HISTORY_SIZE:
                last_scores.popleft()

        screen_binary = pre_process(obv)

        if last_state is None:
            last_state = np.stack(tuple(screen_binary for _ in range(STATE_FRAMES)), axis=2)
            next_action = action_from_output(last_output)
        else:
            screen_binary = np.reshape(screen_binary, (SCREEN_W, SCREEN_H, 1))
            current_state = np.append(last_state[:, :, 1:], screen_binary, axis=2)

            # store the transition in previous_observations
            observations.append((last_state, last_output, reward, current_state, terminal))

            if len(observations) > OBV_SAMPLES_SIZE:
                observations.popleft()

            # only train if done observing
            if len(observations) > INIT_OBV_STEPS:
                train()
                time += 1

            # update the old values
            last_state = current_state

            last_output = select_action(last_state, explore_rate)
            next_action = action_from_output(last_output)

            # gradually reduce the probability of a random action
            if explore_rate > MIN_EXPLORE_RATE  and len(observations) > INIT_OBV_STEPS:
                explore_rate -= (EXPLORE_RATE_0 - MIN_EXPLORE_RATE) /  NUM_EXPLORE_STEPS

            print("Time: %s random_action_prob: %s reward %s scores differential %s" %
                  (time, explore_rate, reward, sum(last_scores) / SCORE_HISTORY_SIZE))

with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    #saver = tf.train.Saver()

    simulation_begin()

