import gym
import numpy as np
import random
import math
from time import sleep

# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

# Defining the environment related constants
NUM_STATE = 10
NUM_ACTION = env.action_space.n
ANGLE_STATE_INDEX = 2
ANGLE_BOUND = (-env.theta_threshold_radians*1.2, env.theta_threshold_radians*1.2)
MIN_EXPLORE_RATE = 0.1
MIN_LEARNING_RATE = 0.1

# Defining the simulation related constants
NUM_EPISODES = 400
MAX_T = 500
DEBUG_MODE = True
ENABLE_UPLOAD = False

# Creating a Q-Table for each action and state pair
q_table = np.zeros((NUM_STATE, NUM_ACTION))

# Manually inputting the preferred Q-values
for state_row in range(len(q_table)):
    if state_row  <= len(q_table)/2:
        q_table[state_row][0] = 1
    else:
        q_table[state_row][1] = 1


def simulate():

    for episode in range(NUM_EPISODES):
        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = angle_to_bucket(obv[ANGLE_STATE_INDEX])
        t_done = 0

        for t in range(MAX_T):
            env.render()

            # Select an action
            action = select_action(state_0, 0)

            # Execute the action
            obv, reward, done, info = env.step(action)

            # Observe the result
            angle = obv[ANGLE_STATE_INDEX]
            state = angle_to_bucket(angle)

            # Setting up for the next iteration
            state_0 = state
            if not done:
                t_done = t

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("Angle: %f" % angle)
                print("State: %d" % state)
                print("Reward: %f" % reward)
                print("Q Table")
                q_table_t = np.transpose(q_table)
                for col in q_table_t:
                    for q in col:
                        print("%.2f" % q, end="  ")
                    print("")

            if angle < -env.theta_threshold_radians*2 or angle > env.theta_threshold_radians*2 :
               print("Episode %d finished after %f time steps" % (episode, t_done))
               break

            #sleep(0.25)


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
