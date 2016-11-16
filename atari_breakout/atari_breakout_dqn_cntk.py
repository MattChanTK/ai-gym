import random
import numpy as np
import math
from time import perf_counter
import os
import sys
from collections import deque

import gym
import cntk
from cntk.layers import Dense
from cntk.models import Sequential, LayerStack


env = gym.make("Breakout-v0")

NUM_ACTIONS = env.action_space.n
SCREEN_H_ORIG, SCREEN_W_ORIG, NUM_COLOUR_CHANNELS = env.observation_space.shape


def preprocess_image(screen_image):

    # crop the top and bottom
    screen_image = screen_image[35:195]

    # down sample by a factor of 2
    screen_image = screen_image[::2, ::2]

    # convert to grey scale
    grey_image = np.zeros(screen_image.shape[0:2])
    for i in range(len(screen_image)):
        for j in range(len(screen_image[i])):
            grey_image[i][j] = np.mean(screen_image[i][j])

    return grey_image.astype(np.float)


IMAGE_H, IMAGE_W = preprocess_image(np.zeros((SCREEN_H_ORIG, SCREEN_W_ORIG))).shape
STATE_DIMS = (IMAGE_H, IMAGE_W)

class Brain:

    BATCH_SIZE = 50

    def __init__(self):

        #### Construct the model ####
        observation = cntk.ops.input_variable(STATE_DIMS, np.float32, name="s")
        q_target = cntk.ops.input_variable(NUM_ACTIONS, np.float32, name="q")

        # Define the structure of the neural network
        self.model = self.create_multi_layer_neural_network(observation, NUM_ACTIONS, 3)

        #### Define the trainer ####
        self.learning_rate = 0.00025

        self.loss =  cntk.ops.reduce_mean(cntk.ops.square(self.model - q_target), axis=0)
        mean_error = cntk.ops.reduce_mean(cntk.ops.square(self.model - q_target), axis=0)

        learner = cntk.adam_sgd(self.model.parameters, self.learning_rate/self.BATCH_SIZE, momentum=0.9)
        self.trainer = cntk.Trainer(self.model, self.loss, mean_error, learner)

    def train(self, x, y):
        data = dict(zip(self.loss.arguments, [y, x]))
        self.trainer.train_minibatch(data, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval(s)

    @staticmethod
    def create_multi_layer_neural_network(input_vars, out_dims, num_hidden_layers):

        input_dims = input_vars.shape
        num_hidden_neurons = 128

        hidden_layer = lambda: Dense(num_hidden_neurons, activation=cntk.ops.relu)
        output_layer = Dense(out_dims, activation=None)

        model = Sequential([LayerStack(num_hidden_layers, hidden_layer),
                            output_layer])(input_vars)
        return model


class Memory:

    def __init__(self, capacity):
        self.examplers = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, sample):
        self.examplers.append(sample)

    def get_random_samples(self, num_samples):
        num_samples = min(num_samples, len(self.examplers))
        return random.sample(tuple(self.examplers), num_samples)

    def get_stack(self, start_index, stack_size):
        end_index = len(self.examplers) - stack_size
        if end_index < 0:
            stack = list(self.examplers) + [self.examplers[-1] for _ in range(-end_index)]
        else:
            start_index = min(start_index, end_index)
            stack = [self.examplers[i + start_index] for i in range(stack_size)]
        return np.stack(stack, axis=-1)

    def get_random_stacks(self, num_samples, stack_size):

        start_indices = random.sample(range(len(self.examplers)), num_samples)
        return [self.get_stack(start_index, stack_size) for start_index in start_indices]

    def get_latest_stack(self, stack_size):
        return self.get_stack(len(self.examplers), stack_size)


class Agent:

    MEMORY_CAPACITY = 100000
    DISCOUNT_FACTOR = 0.99
    MAX_EXPLORATION_RATE = 1.0
    MIN_EXPLORATION_RATE = 0.01
    DECAY_RATE = 0.0001

    def __init__(self):
        self.explore_rate = self.MAX_EXPLORATION_RATE
        self.brain = Brain()
        self.memory = Memory(self.MEMORY_CAPACITY)
        self.steps = 0

    def act(self, s):
        if random.random() < self.explore_rate:
            return random.randint(0, NUM_ACTIONS - 1)
        else:

            return np.argmax(self.brain.predict(s))

    def observe(self, sample):
        self.steps += 1
        self.memory.add(sample)

        # Reduces exploration rate linearly
        self.explore_rate = self.MIN_EXPLORATION_RATE + (self.MAX_EXPLORATION_RATE - self.MIN_EXPLORATION_RATE) * math.exp(-self.DECAY_RATE * self.steps)

    def replay(self):
        batch = self.memory.get_random_samples(self.brain.BATCH_SIZE)
        batch_len = len(batch)

        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        no_state = np.zeros(STATE_DIMS)
        resultant_states = np.array([(no_state if sample[3] is None else sample[3]) for sample in batch], dtype=np.float32)

        q_values_batch = self.brain.predict(states)
        future_q_values_batch = self.brain.predict(resultant_states)

        x = np.zeros((batch_len, ) + STATE_DIMS).astype(np.float32)
        y = np.zeros((batch_len, NUM_ACTIONS)).astype(np.float32)

        for i in range(batch_len):
            state, action, reward, resultant_state = batch[i]

            q_values = q_values_batch[0][i]
            if resultant_state is None:
                q_values[action] = reward
            else:
                q_values[action] = reward + self.DISCOUNT_FACTOR * np.amax(future_q_values_batch[0][i])

            x[i] = state
            y[i] = q_values

        self.brain.train(x, y)

    @classmethod
    def action_from_output(cls, output_array):
        return np.argmax(output_array)


def run_simulation(agent, solved_reward_level):
    state = env.reset()
    state = preprocess_image(state)
    total_rewards = 0

    while True:
        #env.render()

        action = agent.act(state.astype(np.float32))

        resultant_state, reward, done, info = env.step(action)
        resultant_state = preprocess_image(resultant_state)

        if done: # terminal state
            resultant_state = None

        agent.observe((state, action, reward, resultant_state))
        agent.replay()

        state = resultant_state
        total_rewards += reward

        if total_rewards > solved_reward_level or done:
            return total_rewards


def test(model_path, num_episodes=10):

    root = cntk.load_model(model_path)
    observation = env.reset()  # reset environment for new episode
    done = False
    for episode in range(num_episodes):
        while not done:
            try:
                env.render()
            except Exception:
                # this might fail on a VM without OpenGL
                pass

            observation = preprocess_image(observation)
            action = np.argmax(root.eval(observation.astype(np.float32)))
            observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()  # reset environment for new episode


if __name__ == "__main__":

    # Ensure we always get the same amount of randomness
    np.random.seed(0)

    GYM_ENABLE_UPLOAD = False
    GYM_VIDEO_PATH = os.path.join(os.getcwd(), "videos", "atari_breakout_dpn_cntk")
    GYM_API_KEY = "sk_93AMQvdmReWCi8pdL4m6Q"

    MAX_NUM_EPISODES = 5000
    STREAK_TO_END = 120
    DONE_REWARD_LEVEL = 100

    TRAINED_MODEL_DIR = os.path.join(os.getcwd(), "trained_models")
    if not os.path.exists(TRAINED_MODEL_DIR):
        os.makedirs(TRAINED_MODEL_DIR)
    TRAINED_MODEL_NAME = "atari_breakout_dpn.mod"

    EPISODES_PER_PRINT_PROGRESS = 1

    if len(sys.argv) < 2 or sys.argv[1] != "test_only":

        if GYM_ENABLE_UPLOAD:
            env.monitor.start(GYM_VIDEO_PATH, force=True)

        agent = Agent()

        episode_number = 0
        num_streaks = 0
        reward_sum = 0
        solved_episode = -1

        training_start_time = perf_counter()

        while episode_number < MAX_NUM_EPISODES:

            # Run the simulation and train the agent
            reward = run_simulation(agent, DONE_REWARD_LEVEL*2)
            reward_sum += reward

            episode_number += 1
            if episode_number % EPISODES_PER_PRINT_PROGRESS == 0:
                t = perf_counter() - training_start_time
                print("(%d s) Episode: %d, Average reward = %f." % (t, episode_number, reward_sum / EPISODES_PER_PRINT_PROGRESS))
                reward_sum = 0

            # It is considered solved when the sum of reward is over 200
            if reward > DONE_REWARD_LEVEL:
                num_streaks += 1
                solved_episode = episode_number
            else:
                num_streaks = 0
                solved_episode = -1

            # It's considered done when it's solved over 120 times consecutively
            if num_streaks > STREAK_TO_END:
                print("Task solved in %d episodes and repeated %d times." % (episode_number, num_streaks))
                break

        agent.brain.model.save_model(os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME), False)

        if GYM_ENABLE_UPLOAD:
            env.monitor.close()
            gym.upload(GYM_VIDEO_PATH, api_key=GYM_API_KEY)

    # testing the model
    test(os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME), num_episodes=10)
