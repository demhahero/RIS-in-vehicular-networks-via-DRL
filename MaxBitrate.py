# https://github.com/nikhilbarhate99/PPO-PyTorch
import datetime
import os
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Environment import Environment
import sys


def main(item, control_bit):

    ############## Hyperparameters ##############


    max_episodes = N        # max training episodes

    update_timestep = 2000      # update policy every n timesteps


    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    env = Environment()

    env.M = item
    env.control_bit = control_bit

    env.initalize()

    # training loop
    for i_episode in range(1, max_episodes + 1):

        counter = 0
        state = env.reset()

        #state = np.asarray(state, dtype=np.float32)
        slot_rewards = 0
        done = False
        while not done:
            timestep += 1
            # Running policy_old:

            ranks = []
            state = np.delete(state, 0)
            for ite in range(env.max_no_vehicles):
                if(state[ite * 3+0] == -1 and state[ite * 3+1] == -1 and state[ite * 3+2] == -1):
                    state[ite * 3 + 1]=1000
                ranks.append(state[ite * 3 + 1])

            action = 0

            lowest_action = 0
            lowest_value = 10000
            for act in env.action_sets:
                res = np.sum(np.multiply(act, ranks))
                if res < lowest_value:
                    lowest_action = action
                    lowest_value = res
                action+=1

            print("state=", str(state))
            print("action=", env.action_sets[lowest_action])
            print("ranks=", ranks)



            reward, state, done = env.step(action=lowest_action)
            slot_rewards += reward

            print("total=", env.total_reward)


            state = np.asarray(state, dtype=np.float32)



            running_reward += reward

            if done:
                DRL_bitrate.append(float("{:.3f}".format(env.total_reward)))
                DRL_jain.append(float("{:.3f}".format(env.jain)))
                break



        avg= 100
        if(i_episode % avg == 0):
            print("\n_________________________________________________________________________________________________")
            print("****Iteration=", i_episode, "\tBitrate=", float("{:.3f}".format(np.mean(DRL_bitrate[i_episode-avg:i_episode])))
                  , "\tJain=", float("{:.3f}".format(np.mean(DRL_jain[i_episode - avg:i_episode])))
                  )
            print("___________________________________________________________________________________________________")

    print(env.total_reward)
    del env

if __name__ == '__main__':

    To_DRL_bitrate = []
    To_DRL_jian = []

    items = [2]

    for x in items:

        DRL_bitrate = []
        DRL_jain = []


        N = 200
        sample = 200
        smoothing = 400


        main(item=100, control_bit=x)


        your_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = "results/result_" + str(your_datetime) + ".txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(str(filename), "w")

        f.write("DRL_bitrate=" + str(DRL_bitrate) + "\n")
        f.write("DRL_jain=" + str(DRL_jain) + "\n")


        f.write("DRL_bitrate_avg=" + str(np.mean(DRL_bitrate[N-sample:N])) + "\n")
        f.write("DRL_jain_avg=" + str(np.mean(DRL_jain[N-sample:N])) + "\n")

        f.close()

        To_DRL_bitrate.append(np.mean(DRL_bitrate[N-sample:N]))
        To_DRL_jian.append(np.mean(DRL_jain[N-sample:N]))

        #rewards_smoothed = pd.Series(DRL_bitrate).rolling(smoothing, min_periods=0).mean()
        #plt.plot(rewards_smoothed, linewidth=1, color="b", label='Both')

    filename = "results/to"+"_result.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(str(filename), "w")
    f.write("To_DQN_bitrate=" + str(To_DRL_bitrate) + "\n")
    f.write("To_DQN_jain=" + str(To_DRL_jian) + "\n")
    print("END")
    f.close()
