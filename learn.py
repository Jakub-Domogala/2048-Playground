import sys
import numpy as np
import math
import random
from envs.game2048Env import Game2048Env

import gym

def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        state = env.reset()
        total_reward = 0

        for t in range(MAX_TRY):

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            state = next_state

            # env.render()

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay


if __name__ == "__main__":
    env = Game2048Env(size=4)
    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).flatten().astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,), dtype=np.int8)
    simulate()