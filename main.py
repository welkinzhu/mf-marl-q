from environment import GridAgentEnv
from copy import deepcopy
import os, sys, time

EPISODES = 100
# TEST = 10
STEPS = 1000


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = GridAgentEnv()
    # output = open('./log/log.txt', 'w')
    outQtable = open('./log/qtable.txt', 'w')
    outNtable = open('./log/ntable.txt', 'w')
    outVtable = open('./log/vtable.txt', 'w')
    outReward = open('./log/reward.txt', 'w')
    now = time.strftime("%m%d%H%M%S", time.localtime(time.time()))
    outRewardSim = open('./log/reward_simply_' + now + r'.txt', 'w')
    # output.write('episode\tstep\tstate_n\taction_n\treward_n\tdone_n\tarrival_n\n')
    for episode in range(EPISODES):
        state_n = env.n_reset()
        total_reward = 0
        budget = 5000
        total_reward = [0 for _ in range(env.num_nodes)]
        for step in range(STEPS):
            copy_obs_n = deepcopy(state_n)
            # output.write(str(episode) + '\t' + str(step) + '\t' + str(copy_obs_n) + '\t')
            ev_num_before = 0
            for i in range(env.num_nodes):
                ev_num_before += copy_obs_n[i][-1]

            action_n = env.n_choose_action(copy_obs_n)
            # for i in range(env.num_nodes):
            #     output.write(str(env.agents[i].mfq_agent.q_value_action) + '\t' +
            #                  str(env.agents[i].mfq_agent.temp_ratio) + '\t')
            moved_obs_n, arrival_n = env.n_ev_mobility(copy_obs_n, action_n)
            next_state_n, reward_n, done_n = env.n_step(moved_obs_n, action_n, arrival_n)
            total_reward = [total_reward[i] + reward_n[i] for i in range(env.num_nodes)]
            # output.write(str(action_n) + '\t' + str(reward_n) + '\t' + str(done_n) + '\t')
            # output.write(str(arrival_n) + '\t')
            # for i in range(env.num_nodes):
            #     output.write(str(env.agents[i].budget) + '\t')
            # for i in range(env.num_nodes):
            #     output.write(str(env.agents[i].budget) + '\t')
            ev_num_after = 0
            for i in range(env.num_nodes):
                ev_num_after += moved_obs_n[i][-1]

            # output.write(str(ev_num_before) + '\t')
            # output.write(str(ev_num_after) + '\t')
            sum_arrival_n = sum(len(arr) for arr in arrival_n)
            # output.write(str(sum_arrival_n) + '\t')
            # output.write(str(ev_num_before - ev_num_after - sum_arrival_n) + '\t')
            # output.write('\n')
            state_n = next_state_n
            if True in done_n or step == STEPS - 1:
                outReward.write(str(episode) + '\t' + str(step) + '\t')
                outRewardSim.write(str(episode) + '\t' + str(step) + '\t')
                for i in range(env.num_nodes):
                    outReward.write(str(env.agents[i].budget) + '\t')
                outReward.write(str(total_reward) + '\n')
                sum_total_reward = sum(reward_i for reward_i in total_reward)
                outRewardSim.write(str(sum_total_reward) + '\n')
                break
        print(episode)
        print(total_reward)
    # for i in range(env.num_nodes):
    outQtable.write(env.mfq_agent.q_table.to_string())
    outNtable.write(env.mfq_agent.n_table.to_string())
    outVtable.write(env.mfq_agent.v_table.to_string())

    # outQtable.write()
