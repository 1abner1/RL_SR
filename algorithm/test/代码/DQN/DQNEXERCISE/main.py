import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dqn_agent import Agent
import datetime
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape:',env.observation_space.shape)
    print('Number of actions:',env.action_space.n)
    now = datetime.datetime.now()
    date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    writer = SummaryWriter(log_dir="runs/dqn"+"LunarLander-v2"+"_"+date_time)
    score = []
    max_episode = 20000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 0.995
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    agent = Agent(state_size=8,action_size=4,seed=0)
    start_time = datetime.datetime.now()
    for i in range(1,max_episode+1):

        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            print(action)
            env.render()
            next_state,reward,done,_ = env.step(action)
            agent.step(state,action,reward,next_state,done,i,writer)
            state = next_state
            score += reward
            if done:
                writer.add_scalar("reward",score,i)
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end,eps_decay*eps)
        print('\rEpisode {} \tAverage Score: {:.2f}'.format(i,np.mean(scores_window)),end="")
        writer.add_scalar("average_reward",np.mean(scores_window),(datetime.datetime.now()-start_time).total_seconds())
        if i %100 ==0:
            print('\rEpisode{} \tAverage Score:{:.2f}'.format(i,np.mean(scores_window)))
        if i %1000 == 0:
            print("current epslion:",eps)
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score:{:.2f}'.format(i,np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
            break



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
