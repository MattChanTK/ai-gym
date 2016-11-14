import random
import numpy as np
import math
from time import perf_counter
import os
import sys
from collections import deque

import gym
import cntk
from cntk.layers import Dense, Dropout
from cntk.models import Sequential, LayerStack
from cntk.initializer import gaussian


env = gym.make('CartPole-v0')

STATE_DIM  = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n

'''

The neural network and model
'''
class Brain:

    BATCH_SIZE = 50

    def __init__(self):

        #### Construct the model ####
        observation = cntk.ops.input_variable(STATE_DIM, np.float32, name="s")
        q_target = cntk.ops.input_variable(NUM_ACTIONS, np.float32, name="q")

        # Define the structure of the neural network
        self.model = self.create_multi_layers_neural_network(observation, NUM_ACTIONS, 3)
        # self.model = self.create_single_layer_neural_network(observation, NUM_ACTIONS)

        #### Define the trainer ####
        self.learning_rate = 0.00025

        self.loss =  cntk.ops.reduce_mean(cntk.ops.square(self.model - q_target), axis=0)
        mean_error = cntk.ops.reduce_mean(cntk.ops.square(self.model - q_target), axis=0)

        learner = cntk.adam_sgd(self.model.parameters, self.learning_rate/self.BATCH_SIZE, momentum=0.9, gradient_clipping_threshold_per_sample=10)
        self.trainer = cntk.Trainer(self.model, self.loss, mean_error, learner)

    def train(self, x, y, epoch=1, verbose=0):
        data = dict(zip(self.loss.arguments, [y, x]))
        self.trainer.train_minibatch(data, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval(s)

    @staticmethod
    def create_multi_layers_neural_network(input_vars, out_dims, num_hidden_layers):

        input_dims = input_vars.shape[0]
        num_hidden_neurons = input_dims**3

        hidden_layer = lambda: Dense(num_hidden_neurons, activation=cntk.ops.relu, init=gaussian(), init_bias=0.1)
        output_layer = Dense(out_dims, activation=None, init=gaussian(), init_bias=0.1)

        model = Sequential([LayerStack(num_hidden_layers, hidden_layer),
                            output_layer])(input_vars)
        return model

    @staticmethod
    def create_single_layer_neural_network(input_vars, out_dims, dropout_prob=0.0):
        return Brain.create_multi_layers_neural_network(input_vars, out_dims, 1)


class Memory:  # stored as ( s, a, r, s' )

    def __init__(self, capacity):
        self.examplers = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, sample):
        self.examplers.append(sample)

    def get_random_samples(self, num_samples):
        num_samples = min(num_samples, len(self.examplers))
        return random.sample(tuple(self.examplers), num_samples)


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
        batchLen = len(batch)

        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        no_state = np.zeros(STATE_DIM)
        resultant_states = np.array([(no_state if sample[3] is None else sample[3]) for sample in batch], dtype=np.float32)

        q_values_batch = self.brain.predict(states)
        future_q_values_batch = self.brain.predict(resultant_states)

        x = np.zeros((batchLen, STATE_DIM)).astype(np.float32)
        y = np.zeros((batchLen, NUM_ACTIONS)).astype(np.float32)

        for i in range(batchLen):
            state, action, reward, resultant_state = batch[i]

            q_values = q_values_batch[0][i]
            if resultant_state is None:
                q_values[action] = reward
            else:
                q_values[action] = reward + self.DISCOUNT_FACTOR * np.amax(future_q_values_batch[0][i])

            x[i] = state
            y[i] = q_values

        self.brain.train(x, y)


def run_simulation(agent, solved_reward_level):
    state = env.reset()
    total_rewards = 0

    while True:
        # env.render()
        action = agent.act(state.astype(np.float32))

        resultant_state, reward, done, info = env.step(action)

        if done: # terminal state
            resultant_state = None

        agent.observe((state, action, reward, resultant_state))
        agent.replay()

        state = resultant_state
        total_rewards += reward

        if total_rewards / agent.brain.BATCH_SIZE > DONE_REWARD_LEVEL or done:
            return total_rewards


def test(model_path, num_episodes=10):

    root = cntk.load_model(model_path)
    observation = env.reset()  # reset environment for new episode
    done = False
    for episode in range(num_episodes):
        while not done:
            try:
                env.render()
            except Exception as e:
                print(e)
            action = np.argmax(root.eval(observation.astype(np.float32)))
            observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()  # reset environment for new episode


if __name__ == "__main__":

    GYM_ENABLE_UPLOAD = False
    GYM_VIDEO_PATH = os.path.join(os.getcwd(), "videos", "cart_pole_dpn_cntk")
    GYM_API_KEY = "sk_93AMQvdmReWCi8pdL4m6Q"

    MAX_NUM_EPISODES = 5000
    STREAK_TO_END = 120
    DONE_REWARD_LEVEL = 200

    TRAINED_MODEL_DIR = os.path.join(os.getcwd(), "trained_models")
    if not os.path.exists(TRAINED_MODEL_DIR):
        os.makedirs(TRAINED_MODEL_DIR)
    TRAINED_MODEL_NAME = "cart_pole_dpn.mod"

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
            if episode_number % agent.brain.BATCH_SIZE == 0:
                t = perf_counter() - training_start_time
                print("(%d s) Episode: %d, Average reward = %f." % (t, episode_number, reward_sum / agent.brain.BATCH_SIZE))

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

                reward_sum = 0

        agent.brain.model.save_model(os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME), False)

        if GYM_ENABLE_UPLOAD:
            env.monitor.close()
            gym.upload(GYM_VIDEO_PATH, api_key=GYM_API_KEY)

    # testing the model
    test(os.path.join(TRAINED_MODEL_DIR, TRAINED_MODEL_NAME), num_episodes=10)
