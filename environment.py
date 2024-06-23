import numpy as np
import pandas as pd
import random
from graph import GridAgent
import networkx as nx
from mf_q import MFQLearning


class GridAgentEnv(object):
    def __init__(self):

        self.load_ran_ln = []
        self.ideal_load_ran_ln = []
        self.renewable_rand_ln = []
        self.agents = []
        self.graph = nx.Graph()
        self.num_nodes = 0
        self.nodes = []
        self.shortest_path = []
        self.nbr_ave_n = []
        self.nbr_obs_ave_n = []
        self.action_space = list(range(1, 101, 10))
        self.mfq_agent = MFQLearning(self.action_space)
        self.obs_space = []
        self.simply_obs_space = []
        self.action_space = []
        self.reward_space = []
        self.next_obs_space = []
        self.simply_next_obs_space = []

    def n_choose_action(self, obs_n):
        # find ave mean action
        nbr_obs_ave_n = [[0, 0, 0] for _ in range(self.num_nodes)]
        self.obs_space = obs_n
        self.simply_obs_space = self.n_simply_obs(obs_n)
        obs_n_sum = [0.0 for _ in range(len(self.simply_obs_space[0]))]
        for i in range(len(self.simply_obs_space)):
            for j in range(len(self.simply_obs_space[i])):
                obs_n_sum[j] += self.simply_obs_space[i][j]
        obs_n_ave = [obs_sum/len(self.simply_obs_space) for obs_sum in obs_n_sum]
        for i in range(self.num_nodes):
            obs_i_ave = [0.0 for _ in range(len(obs_n_ave))]
            for j in range(len(obs_n_ave)):
                obs_i_ave[j] = obs_n_ave[j] * (1 + 1/(self.num_nodes - 1)) - self.simply_obs_space[i][j]
            nbr_obs_ave_n[i] = obs_i_ave
        action_n = []
        for i in range(self.num_nodes):
            action = self.mfq_agent.choose_action(self.simply_obs_space[i], self.nbr_obs_ave_n[i], self.nbr_ave_n[i])
            action_n.append(action)
        for i in self.graph.nodes:
            nbr_act_s = []
            i_adj = self.graph.adj[i]
            for nbr in self.graph.adj[i]:   # nbr的范围是1-5
                nbr_act_s.append(action_n[nbr-1])  # 对应的action_n就是0-4
            self.nbr_ave_n[i-1] = sum(nbr_act_s) / len(nbr_act_s)  # 更新平均动作
        return action_n

    def n_step(self, moved_obs_n, action_n, arrival_n):
        next_obs_n = []  # 保留step函数后的
        reward_n = []
        done_n = []
        for i in range(self.num_nodes):
            obs, reward, done = self.agents[i].step_state(moved_obs_n[i], action_n[i], arrival_n[i])
            next_obs_n.append(obs)
            reward_n.append(reward)
            done_n.append(done)
        self.next_obs_space = next_obs_n
        self.simply_next_obs_space = self.n_simply_obs(next_obs_n)
        self.action_space = action_n
        self.reward_space = reward_n
        self.n_update()
        return next_obs_n, reward_n, done_n

    def n_update(self):
        # self, state, nbr_ave, action, reward, next_state
        for i in range(self.num_nodes):
            self.mfq_agent.update(self.simply_obs_space[i], self.nbr_obs_ave_n[i], self.nbr_ave_n[i], self.action_space[i],
                                  self.reward_space[i], self.simply_next_obs_space[i])

    def n_reset(self):
        self.n_graph_init()
        obs_n = []
        self.agents = []
        self.load_ran_ln = []
        self.ideal_load_ran_ln = []
        self.renewable_rand_ln = []
        for i in range(self.num_nodes):
            load_rand = random.uniform(0.5, 2)  # 之前是2
            # load_rand = 3
            self.load_ran_ln.append(load_rand)
            ideal_rand = random.uniform(load_rand - 0.1, load_rand + 0.1)
            self.ideal_load_ran_ln.append(ideal_rand)
            renewable_rand = random.uniform(load_rand - 0.4, load_rand + 0.4)
            self.renewable_rand_ln.append(renewable_rand)
            agent = GridAgent(load_rand, ideal_rand, renewable_rand)
            obs = agent.reset()
            # agent.budget = 10000
            self.agents.append(agent)
            obs_n.append(obs)
        return obs_n

    def n_graph_init(self):
        self.graph.add_edge(1, 2, weight=1)
        self.graph.add_edge(1, 3, weight=1)
        self.graph.add_edge(2, 4, weight=1)
        self.graph.add_edge(3, 4, weight=1)
        self.graph.add_edge(2, 3, weight=1)
        self.graph.add_edge(1, 5, weight=1)
        self.graph.add_edge(5, 4, weight=1)
        # 然后增加5个agent
        # self.graph.add_edge(5, 6, weight=1)
        # self.graph.add_edge(5, 7, weight=1)
        # self.graph.add_edge(6, 7, weight=1)
        # self.graph.add_edge(6, 8, weight=1)
        # self.graph.add_edge(1, 8, weight=1)
        # self.graph.add_edge(7, 8, weight=1)  # 加了一个五宫格
        # self.graph.add_edge(6, 9, weight=1)
        # self.graph.add_edge(9, 10, weight=1)
        # self.graph.add_edge(4, 10, weight=1)
        # self.graph.add_edge(5, 10, weight=1)
        # self.graph.add_edge(4, 9, weight=1)

        # 再增加5个agent
        # self.graph.add_edge(8, 11, weight=1)
        # self.graph.add_edge(11, 12, weight=1)
        # self.graph.add_edge(2, 12, weight=1)
        # self.graph.add_edge(1, 11, weight=1)
        # self.graph.add_edge(1, 12, weight=1)    # 增加5个
        # self.graph.add_edge(12, 13, weight=1)
        # self.graph.add_edge(13, 14, weight=1)
        # self.graph.add_edge(14, 3, weight=1)
        # self.graph.add_edge(3, 13, weight=1)
        # self.graph.add_edge(12, 14, weight=1)  # 增加5个
        # self.graph.add_edge(7, 15, weight=1)
        # self.graph.add_edge(11, 15, weight=1)
        # self.graph.add_edge(8, 15, weight=1)

        #
        # self.graph.add_edge(1, 2, weight=1)
        # self.graph.add_edge(1, 3, weight=1)
        # self.graph.add_edge(1, 4, weight=1)
        # self.graph.add_edge(1, 5, weight=1)

        # n = len(self.graph.nodes)
        self.num_nodes = len(list(self.graph.nodes))
        self.nodes = list(self.graph.nodes)
        self.nbr_ave_n = [1.0 for _ in range(self.num_nodes)]
        self.nbr_obs_ave_n = [[0, 0, 0] for _ in range(self.num_nodes)]
        # shortest_path = [[n for _ in range(n)] for _ in range(n)]
        # node_list = list(self.graph.nodes)
        # for i in range(n):
        #     shortest_path[i][i] = 0
        #     for j in range(i):
        #         shortest_path_length = nx.shortest_path_length(G, source=node_list[i], target=node_list[j],
        #                                                        weight="weight")
        #         if shortest_path_length < shortest_path[i][j]:
        #             shortest_path[i][j] = shortest_path_length
        #             shortest_path[j][i] = shortest_path_length
        # self.shortest_path = shortest_path

    def n_ev_mobility(self, obs_n, action_n):
        prices_n = [0 for _ in range(self.num_nodes)]
        prob_n = []
        nbr_n = []
        for i in range(self.num_nodes):
            time, load, renew_energy, ideal_load, soc_s, ev_num = obs_n[i]
            if ev_num > 0:
                prices_n[i] = action_n[i] / ev_num
            else:
                prices_n[i] = action_n[i]
        for i in self.graph.nodes:
            adj_i = self.graph.adj[i]
            prob_s = []
            nbr_s = []
            for nbr in adj_i:
                dis = adj_i[nbr]['weight']
                gap = prices_n[nbr-1] - prices_n[i-1]
                prob = 0.3 * gap * 0.1 / dis  # 现在是0.1，改动后是1？
                if prob < 0:
                    prob = 0
                prob_s.append(prob)
                nbr_s.append(nbr)
            prob_s.append(1-sum(prob_s))
            nbr_s.append(-1)
            prob_n.append(prob_s)
            nbr_n.append(nbr_s)
        # departure_n = {}
        arrival_n = [[] for _ in range(self.num_nodes)]
        new_obs_n = [[] for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            # if self.nodes[i] not in departure_n:
            #     departure_n[self.nodes[i]] = {}
            time, load, renew_energy, ideal_load, soc_s, ev_num = obs_n[i]
            now_soc_s = []

            for soc in soc_s:
                mvd_ev_num = 0
                dist = random.choices(nbr_n[i], weights=prob_n[i], k=1)[0]
                if dist != -1 and mvd_ev_num <= ev_num/10:
                    arrival_n[dist - 1].append(soc)
                    mvd_ev_num += 1
                elif dist == -1:   # 还留在原地
                    now_soc_s.append(soc)
                    # if dist not in departure_n[self.nodes[i]]:
                    #     departure_n[self.nodes[i]][dist] = []
                    # departure_n[self.nodes[i]][dist].append(soc)
            new_obs_n[i] = (time, load, renew_energy, ideal_load, now_soc_s, len(now_soc_s))
        # print(1)
        return new_obs_n, arrival_n

    def n_simply_obs(self, state_n):
        simply_obs_n = []
        for state in state_n:
            time, load, renew_energy, ideal_load, soc_s, ev_num = state
            # load = load // 10 * 10
            # renew_energy = renew_energy // 10 * 10
            soc_sum = sum(soc_s) // 30 * 30
            load_gap = (ideal_load - load) // 30 * 30
            renew_energy = renew_energy // 20 * 20
            ev_num = ev_num // 5 * 5
            # obs = (load, renew_energy, ideal_load, soc_sum, ev_num)
            # obs = (load_gap, renew_energy, soc_sum, ev_num)
            obs = [load_gap, renew_energy, soc_sum]
            simply_obs_n.append(obs)
        return simply_obs_n

