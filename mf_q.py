import numpy as np
import pandas as pd
import math
from utils import *
import matplotlib.pyplot as plt
import os
import datetime
import random

# REPLAY_BUFFER_SIZE = 1000000
# REPLAY_START_SIZE = 10000
# BATCH_SIZE = 64
GAMMA = 1
H = 500
B = 500
# actions = 7
# action_l = 1
# action_h = 26  # 实际为1-25
# action_step = 4  # 每隔4个取一个
x_dual = 1
epsilon = math.sqrt(math.log(10, 3)/B)

iota = 10  # log(SAT/p)


class MFQLearning:

    def __init__(self, action_space):
        # self.environment = env
        # self.action_dim = actions + 1
        self.actions = action_space    # list([0, 1, 5, 9, 13, 17, 21, 25])
        self.action_dim = len(self.actions)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.int)
        self.n_table = pd.DataFrame(columns=self.actions, dtype=np.int)
        self.v_table = pd.DataFrame(columns=[1], dtype=np.int)
        self.q_value_action = []
        self.temp_ratio = []
        self.gamma = 0.9
        self.lr = 0.01

    def choose_action(self, observation, nbr_obs, nbr_ave):
        # Checking if the state exists in the table
        observation = self.simply_obs(observation)
        nbr_obs = self.simply_obs(nbr_obs)
        self.check_obs_nbr_exist(observation, nbr_obs, nbr_ave)
        self.check_obs_exist(observation, nbr_obs)
        temp = []
        i = 0
        con_state = observation + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        for act in self.actions:
            t = self.q_table.loc[obs_nbr, act]
            # print(t)
            t = t / (x_dual * (act + 0.00001))
            temp.append(t)
        self.q_value_action = self.q_table.loc[obs_nbr, :]
        self.temp_ratio = temp
        zipped_ln = zip(temp, self.actions[:])
        sorted_lists = sorted(zipped_ln)
        sorted_tmp, sorted_act = [list(tup) for tup in zip(*sorted_lists)]
        # a = temp.index(max(temp))
        action = sorted_act[-1]
        return action

    def q_learning_action(self, observation, nbr_obs, nbr_ave):
        observation = self.simply_obs(observation)
        nbr_obs = self.simply_obs(nbr_obs)
        self.check_obs_nbr_exist(observation, nbr_obs, nbr_ave)
        self.check_obs_exist(observation, nbr_obs)
        temp = []
        i = 0
        con_state = observation + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        for act in self.actions:
            t = self.q_table.loc[obs_nbr, act]
            temp.append(t)
        self.q_value_action = self.q_table.loc[obs_nbr, :]
        self.temp_ratio = temp
        zipped_ln = zip(temp, self.actions[:])
        sorted_lists = sorted(zipped_ln)
        sorted_tmp, sorted_act = [list(tup) for tup in zip(*sorted_lists)]
        # a = temp.index(max(temp))
        action = sorted_act[-1]
        return action

    def bandit_action(self, observation, nbr_obs, nbr_ave):
        observation = self.simply_obs(observation)
        nbr_obs = self.simply_obs(nbr_obs)
        self.check_obs_nbr_exist(observation, nbr_obs, nbr_ave)
        self.check_obs_exist(observation, nbr_obs)
        temp = []
        i = 0
        con_state = observation + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        if np.random.uniform() < 0.9:
            state_action = self.q_table.loc[obs_nbr, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def random_action(self, observation, nbr_obs, nbr_ave):
        action = np.random.choice(self.actions)
        return action

    def greedy_action(self, observation, nbr_obs, nbr_ave):
        action = self.actions[-1]
        return action

    # Function for learning and updating Q-table with new knowledge
    def update(self, state, nbr_obs, nbr_ave, action, reward, next_state):
        # Checking if the next step exists in the Q-table
        state = self.simply_obs(state)
        nbr_obs = self.simply_obs(nbr_obs)
        next_state = self.simply_obs(next_state)
        self.check_obs_exist(next_state, nbr_obs)
        con_state = state + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        self.check_obs_nbr_exist(state, nbr_obs, nbr_ave)
        self.n_table.loc[obs_nbr, action] += 1
        n = int(self.n_table.loc[obs_nbr, action])
        alpha = (H + 1) / (H + n)
        bonus = math.sqrt(H ** 3 * iota / n) / 10000
        q_value = int(self.q_table.loc[obs_nbr, action])
        # print('previous q: %d, alpha: %f, bonus: %f' %(q_value, alpha, bonus))
        v_value = int(self.v_table.loc[str(next_state + nbr_obs), 0])
        q_value = int((1 - alpha) * q_value + alpha * (reward + v_value + bonus))
        # print('updated q is %d' %(q_value))
        self.q_table.loc[obs_nbr, action] = int(q_value)
        self.v_table.loc[str(state + nbr_obs)] = min(H, int(self.q_table.loc[str(obs_nbr), action]))
        # Current state in the current position
        # return self.q_table.loc[state, action]

    # Function for learning and updating Q-table with new knowledge
    def update_q(self, state, nbr_obs, nbr_ave, action, reward, next_state):
        state = self.simply_obs(state)
        nbr_obs = self.simply_obs(nbr_obs)
        next_state = self.simply_obs(next_state)
        self.check_obs_exist(next_state, nbr_obs)
        con_state = state + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        self.check_obs_nbr_exist(state, nbr_obs, nbr_ave)
        self.n_table.loc[obs_nbr, action] += 1
        # Current state in the current position
        q_predict = self.q_table.loc[obs_nbr, action]
        v_value = int(self.v_table.loc[str(next_state + nbr_obs), 0])
        q_target = reward + self.gamma * v_value
        # Updating Q-table with new knowledge
        self.q_table.loc[obs_nbr, action] += self.lr * (q_target - q_predict)
        self.v_table.loc[str(state + nbr_obs)] = min(H, int(self.q_table.loc[str(obs_nbr), action]))
        # return self.q_table.loc[state, action]

    def update_bandit(self, state, nbr_obs, nbr_ave, action, reward, next_state):
        state = self.simply_obs(state)
        nbr_obs = self.simply_obs(nbr_obs)
        con_state = state + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        self.n_table.loc[obs_nbr, action] += 1
        n = int(self.n_table.loc[obs_nbr, action])
        q_value = int(self.q_table.loc[obs_nbr, action])
        # print('previous q: %d, alpha: %f, bonus: %f' %(q_value, alpha, bonus))
        q_value = (n * q_value + reward) / (n + 1)
        # print('updated q is %d' %(q_value))
        self.q_table.loc[obs_nbr, action] = q_value

    # 如果没有visit到，就加上
    def check_obs_nbr_exist(self, state, nbr_obs, nbr_ave):
        con_state = state + nbr_obs
        obs_nbr = str(np.append(con_state, nbr_ave))
        if obs_nbr not in self.q_table.index:
            new_q = pd.Series([H] * self.action_dim, index=self.q_table.columns, name=obs_nbr,)
            self.q_table = self.q_table.append(new_q)
        if obs_nbr not in self.n_table.index:
            new_n = pd.Series([0] * self.action_dim, index=self.q_table.columns, name=obs_nbr,)
            self.n_table = self.n_table.append(new_n)

    # 检查v表中，是否有该state
    def check_obs_exist(self, state, nbr_obs):
        con_state = state + nbr_obs
        state = str(con_state)
        if state not in self.v_table.index:
            new_v = pd.Series([0], name=state, )
            self.v_table = self.v_table.append(new_v)

    # 简化状态
    def simply_obs(self, state):
        load_gap, renew_energy, soc_sum = state
        load_gap = load_gap // 30 * 30
        renew_energy = renew_energy // 20 * 20
        soc_sum = soc_sum // 5 * 5
        obs = (load_gap, renew_energy, soc_sum)
        return obs

    def final(self):
        outqtablepath = './table/qtable.xlsx'
        self.q_table.to_excel(outqtablepath, index=True, header=True)
        outvtablepath = './table/vtable.xlsx'
        self.v_table.to_excel(outvtablepath, index=True, header=True)
        outntablepath = './table/ntable.xlsx'
        self.n_table.to_excel(outntablepath, index=True, header=True)



