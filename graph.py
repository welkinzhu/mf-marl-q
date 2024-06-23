import numpy as np
from utils import *
from mf_q import MFQLearning
import random
# class Vertex:
#     def __init__(self, key):
#         self.id = key
#         self.connectedTo = {}
#
#     # 从这个顶点添加一个连接到另一个
#     def addNeighbor(self, nbr, weight=0):  # nbr是顶点对象的key
#         self.connectedTo[nbr] = weight
#     # 顶点数据字符串化，方便打印
#
#     def __str__(self):
#         return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])
#         # 返回邻接表中的所有顶点
#
#     def getConnections(self):
#         return self.connectedTo.keys()
#
#     # 返回key
#     def getId(self):
#         return self.id
#
#     # 返回顶点边的权重
#     def getWeight(self, nbr):
#         return self.connectedTo[nbr]
#
#
# class Graph:
#     def __init__(self):
#         self.vertList = {}
#         self.numVertices = 0
#
#     # 新加节点
#     def addVertex(self, key):
#         self.vertList = self.numVertices + 1
#         new_vertex = Vertex(key)
#         self.vertList[key] = new_vertex
#
#     # 通过key查找顶点
#     def getVertex(self, n):
#         if n in self.vertList:
#             return self.vertList[n]
#         else:
#             return None
#
#     def __contains__(self, n):
#         return n in self.vertList
#
#     def addEdge(self, f, t, cost=0):
#         if f not in self.vertList:  # 不存在的顶点先添加
#             nv = self.addVertex(f)
#         if t not in self.vertList:
#             nv = self.addVertex(t)
#         self.vertList[f].addNeighbor(self.vertList[t], cost)
#
#     def getVertices(self):
#         return self.vertList.keys()
#
#     def __iter__(self):
#         return iter(self.vertList.values())

# g = Graph()
# for i in range(6):
#     g.addVertex(i)
# g.addEdge(0, 1, 5)
# g.addEdge(0, 5, 2)
# g.addEdge(1, 2, 4)
# g.addEdge(2, 3, 9)
# g.addEdge(3, 4, 7)
# g.addEdge(3, 5, 3)
# g.addEdge(4, 0, 1)
# g.addEdge(5, 4, 8)
# g.addEdge(5, 2, 1)
# for v in g:  # 遍历输出
#     for w in v.getConnections():
#         print("( %s , %s )" % (v.getId(), w.getId()))
# import networkx as nx

# G = nx.Graph()
# G.add_edge(1, 2, weight=1)
# G.add_edge(1, 3, weight=1)
# G.add_edge(2, 4, weight=1)
# G.add_edge(3, 4, weight=1)
# G.add_edge(2, 3, weight=1)
# G.add_edge(1, 5, weight=1)
# G.add_edge(5, 4, weight=1)
#
# for i in G.nodes:
#     adj_i = G.adj[i]
#     for nbr in adj_i:
#         dis = adj_i[nbr]
#         print(dis)
#         print(dis['weight'])
# for i in G.adj[1]:
#     print(i)
# print(1)
# G.add_node("A")
# G.add_node("B")
# G.add_node("C")
# G.add_node("D")
# G.add_node("D")


class GridAgent(object):
    def __init__(self, load_rate, ideal_rate, renewable_rate):
        super(GridAgent, self).__init__()
        self.obs_high = np.array([24, 1000, 1000, 40000, 100], dtype=np.int)
        self.obs_low = np.array([0, 0, 0, 0, 0], dtype=np.int)
        self.load_arr = [363, 360, 350, 349, 342, 331, 327, 367, 428, 496, 503, 496,
                         487, 475, 470, 455, 437, 427, 403, 402, 401, 394, 390, 376]
        self.renewable_arr = [12, 0, 1, 8, 19, 34, 95, 118, 140, 68, 31, 40,
                              44, 81, 116, 118, 103, 68, 36, 10, 3, 0, 2, 19]
        self.load_rate = load_rate
        self.ideal_rate = ideal_rate
        self.renewable_rate = renewable_rate
        self.budget = 3000
        self.done = False
        self.action_space = list(range(1, 1001, 50))
        self.n_actions = len(self.action_space)
        self.mfq_agent = MFQLearning(self.action_space)

    def step_state(self, state, action, arrival):
        time, load, renew_energy, ideal_load, soc_s, ev_num = state
        # done = False
        new_time = time_step(time)
        new_load = load_step(new_time) * self.load_rate  # load是当前区域的load
        new_renew_energy = renewable_step(new_time) * self.renewable_rate   # 可再生能源也是当前区域的
        new_ideal_load = ideal_step(new_time) * self.ideal_rate  # 理想负载是分区域，还是总体的
        sr_matrix = gene_sr(state, action)  # 输出用于DP的sr表
        assign_action = assignment(sr_matrix)  # 分配action
        # assign_action = uniform_assign(state, action)  # 均匀分的
        remain_action, decrease = user_reaction(assign_action, state, action)
        new_soc_pool, new_ev_num = ev_step(state, arrival)
        # 根据给的action，确定多少车辆接受提议
        # new_soc_pool, new_ev_num = arrival_step(soc_s, ev_num, arrival)  # 参与V2G的情况，和离开的情况，更新现在的soc池的信息和ev数量
        reward = calc_reward(load, renew_energy, ideal_load, decrease)  # 根据当前load变化，更新reward
        state = (new_time, new_load, new_renew_energy, new_ideal_load, new_soc_pool, new_ev_num)
        self.budget = self.budget - action + remain_action
        if self.budget <= 0:
            done = True
            # self.budget = 10000
        else:
            done = False
        return state, reward, done

    def reset(self):
        self.budget = 3000
        self.done = False
        time = np.random.randint(0, 23)
        load = self.load_arr[time] * self.load_rate
        renew_energy = self.renewable_arr[time] * self.renewable_rate
        ideal_load = ideal_step(time) * self.ideal_rate
        # renew_load = load - renewable  # 减去可再生能源后的load
        # load_gap = renew_load - ref_load
        ev_num = random.randint(30, 40)  # random生成ev数量
        # ev_num = 35
        soc_s = [random.randint(16, 31) for _ in range(ev_num)]
        # soc_pool = sum(soc_num)  # random() 生成 每辆车的信息
        state = np.array([time, load, renew_energy, ideal_load, soc_s, ev_num])
        return state

