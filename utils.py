import numpy as np


def time_step(time):
    time = time + 1
    if time < 24:
        time = time
    else:
        time = time - 24
    return time


def load_step(time):  # load是当前区域的load
    load_arr = [363, 360, 350, 349, 342, 331, 327, 367, 428, 496, 503, 496,
                487, 475, 470, 455, 437, 427, 403, 402, 401, 394, 390, 376]
    load = load_arr[time]
    load = np.random.normal(loc=load, scale=load / 10, size=None)
    return load


def renewable_step(time):  # 可再生能源也是当前区域的
    renewable_arr = [12, 0, 1, 8, 19, 34, 95, 118, 140, 68, 31, 40,
                     44, 81, 116, 118, 103, 68, 36, 10, 3, 0, 2, 19]
    renewable = renewable_arr[time]
    return renewable


def ideal_step(time):  # 理想负载是分区域，还是总体的
    ideal_arr = [270, 260, 250, 240, 230, 220, 280, 340, 400, 400, 400, 400,
                 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280]
    ideal = ideal_arr[time]
    return ideal


def ev_step(state, arrival):  # 根据给的action，确定多少车辆接受提议，多少去了其他地方
    # return decisions, departure
    # remain = sum(assign_action)
    # decrease = 0
    # if time < 6:
    #     ratio = 0.6
    # else:
    #     ratio = 1.0
    time, load, renew_energy, ideal_load, soc_s, ev_num = state
    # if ev_num > 0:
    #     each_incen = action / ev_num
    # else:
    #     each_incen = action
    # for i in range(len(soc_s)):
    #     gap = int(soc_s[i] * each_incen / 10) * ratio
    #     if 0 < gap <= 3:
    #         if soc_s[i] < 36:
    #             soc_s[i] += 3
    #             remain = remain
    #     elif 3 < gap <= 7:
    #         if soc_s[i] > 3:
    #             soc_s[i] -= 3
    #             decrease += 3
    #             remain = remain - each_incen
    #     elif 7 < gap:
    #         if soc_s[i] > 7:
    #             soc_s[i] -= 7
    #             decrease += 7
    #             remain = remain - each_incen
    new_soc_s = soc_s + arrival
    new_ev_num = ev_num + len(arrival)
    # state = (time, load, renew_energy, ideal_load, soc_s, ev_num)
    return new_soc_s, new_ev_num


def calc_reward(load, renew_energy, ideal_load, decrease):  # 根据当前load变化，更新reward
    if load <= ideal_load:
        reward = 0
    elif ideal_load < load < ideal_load + 0.8 * renew_energy:
        reward = decrease * 0.6
    elif load >= ideal_load + 0.8 * renew_energy:
        reward = min(load - ideal_load - 0.8 * renew_energy, decrease)

    return reward / 100


def uniform_assign(state, action):
    time, load, renew_energy, ideal_load, soc_s, ev_num = state
    if len(soc_s) == 0:
        return []
    else:
        uniform_ln = [action//len(soc_s) for i in range(len(soc_s))]
        return uniform_ln


def assignment(values):
    # print(values)
    # print(type(values))
    n = len(values)  # n个charging pole
    if n > 0:
        m = len(values[0])  # m个价格
    else:
        return values
    dp = [0 for _ in range(m)]
    s = [[0] for _ in range(m)]
    for i in range(n):
        for j in range(m - 1, -1, -1):
            if i == 0:
                dp[j] = values[i][j]
                s[j] = [j * 1]
            else:
                f = 0
                for k in range(j, 0, -1):
                    if dp[j] < dp[j - k] + values[i][k]:
                        f += 1
                        dp[j] = dp[j - k] + values[i][k]
                        s[j - k].append(k * 1)
                        s[j] = s[j - k].copy()
                        s[j - k].pop()
                if f == 0:
                    s[j].append(0)
    # sum_sr = dp[-1]
    # print(s)
    # print(type(s))
    result = s[-1]
    return result


# input:需要的电量向量,1*10，acton确定输出的大小
# output:values matrix, 10*|action|
def gene_sr(state, action):
    time, load, renew_energy, ideal_load, soc_s, ev_num = state
    need_energy = [39-soc_s[i] for i in range(len(soc_s))]
    values = []
    for e in need_energy:
        if action == 0:
            val = [0]
        else:
            val = [0 for i in range(action)]
            for a in range(action):
                if e == 39:  # 如果电多，更大概率接受V2G
                    val[a] = 0
                elif 35 < e <= 39:
                    val[a] = 1 * a + 1
                elif 31 < e <= 35:
                    val[a] = 2 * a + 1
                elif 23 < e <= 31:
                    val[a] = 4 * a + 1
                elif 15 < e <= 23:
                    val[a] = 6 * a + 1
                elif 7 < e <= 15:
                    val[a] = 8 * a + 1
                elif 0 < e <= 7:
                    val[a] = 10 * a + 1
            for a in range(action):
                if val[a] > 50:
                    val[a] = 50
        values.append(val)
    return values


def user_reaction(assign_action, state, action):
    time, load, renew_energy, ideal_load, soc_s, ev_num = state
    need_energy = [39 - soc_s[i] for i in range(len(soc_s))]
    # decision = [0 for i in range(len(soc_s))]
    remain = action
    decrease = 0
    if time < 6:
        ratio = 0.6
    else:
        ratio = 1.0
    # for i in range(len(soc_s)):
    #     gap = int(soc_s[i] * each_incen / 10) * ratio
    #     if 0 < gap <= 3:
    #         if soc_s[i] < 36:
    #             soc_s[i] += 3
    #             remain = remain
    #     elif 3 < gap <= 7:
    #         if soc_s[i] > 3:
    #             soc_s[i] -= 3
    #             decrease += 3
    #             remain = remain - each_incen
    #     elif 7 < gap:
    #         if soc_s[i] > 7:
    #             soc_s[i] -= 7
    #             decrease += 7
    #             remain = remain - each_incen
    for i in range(len(soc_s)):
        gap = int(need_energy[i] * assign_action[i] / 5) * ratio
        if 0 < gap <= 3:  # 不情愿改变时
            if soc_s[i] < 36:   # 还差一点满电，就加3
                soc_s[i] += 3
                remain = remain
            else:
                soc_s[i] = 39  # 充满
        elif 3 < gap <= 7:   # 做出一定改变
            if soc_s[i] > 3:  # 有一定电量
                soc_s[i] -= 3
                decrease += 3
                remain = remain - assign_action[i]
        elif 7 < gap:   # 做出较多改变
            if soc_s[i] > 7:  # 电多
                soc_s[i] -= 7
                decrease += 7
                remain = remain - assign_action[i]

    return remain, decrease
