import gym
import numpy as np
import random
import math
from time import sleep

# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

# Defining the environment related constants
NUM_STATE = 15
NUM_ACTION = env.action_space.n
ANGLE_STATE_INDEX = 2
ANGLE_BOUND = (-env.theta_threshold_radians*2, env.theta_threshold_radians*2)
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

# Defining the simulation related constants
NUM_EPISODES = 5000
MAX_T = 500
DEBUG_MODE = True
ENABLE_UPLOAD = True

# Creating a Q-Table for each action and state pair
q_table = np.zeros((NUM_STATE, NUM_ACTION))

if ENABLE_UPLOAD:
    env.monitor.start('/tmp/cartpole-experiment-1', force=True)


def simulate():
    # Defining the learning related parameters
    learning_rate = 0.2
    explore_rate = 0.8
    discount_factor = 0.999

    for episode in range(NUM_EPISODES):
        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = angle_to_bucket(obv[ANGLE_STATE_INDEX])
        t_done = 0

        for t in range(MAX_T):
            env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, info = env.step(action)

            # Observe the result
            angle = obv[ANGLE_STATE_INDEX]
            state = angle_to_bucket(angle)


            # Update the Q based on the result
            best_q = np.amax(q_table[state,:])
            q_table[state_0, action] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0, action])

            # Setting up for the next iteration
            state_0 = state
            if not done:
                t_done = t

            # Print data
            if (DEBUG_MODE):
                print("\nepisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("Angle: %f" % angle)
                print("State: %d" % state)
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Q Table")
                q_table_t = np.transpose(q_table)
                for col in q_table_t:
                    for q in col:
                        print("%.2f" % q, end="  ")
                    print("")

            if done or angle < -env.theta_threshold_radians*2 or angle > env.theta_threshold_radians*2 :
               print("Episode %d finished after %f time steps" % (episode, t_done))
               break

            #sleep(0.5)

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

    if ENABLE_UPLOAD:
        env.monitor.close()
        gym.upload('/tmp/cartpole-experiment-1',
                   api_key='sk_93AMQvdmReWCi8pdL4m6Q')


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state,:])
    return action


def get_explore_rate(t):
    return min(1, MIN_EXPLORE_RATE*NUM_EPISODES/(t+1))

def get_learning_rate(t):
    return min(0.8, MIN_LEARNING_RATE*NUM_EPISODES/(t+1))


def angle_to_bucket(angle):
    angle = min(ANGLE_BOUND[1], max(ANGLE_BOUND[0], angle))
    return int((angle - ANGLE_BOUND[0])/(ANGLE_BOUND[1] - ANGLE_BOUND[0])*(NUM_STATE-1))


def bucket_to_angle(state):

    return state / (NUM_STATE-1) * (ANGLE_BOUND[1] - ANGLE_BOUND[0]) + ANGLE_BOUND[0]


if __name__ == "__main__":
    simulate()
