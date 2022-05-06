# This is a sample Python script.
import numpy as np
import gym
import time
from collections import namedtuple
from collections import deque
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class PG_nn(nn.Module):
    def __init__(self,input_shape,n_actions):
        super(PG_nn,self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(input_shape[0],64),
            nn.ReLU(),
            nn.Linear(64,n_actions)
        )
    def forward(self,x):
        return self.mlp(x.float())
def discounted_rewards(memories,gamma):
    disc_rew=np.zeros(len(memories))
    run_add=0
    for t in reversed(range(len(memories))):
        if memories[t].done:run_add=0
        run_add=run_add*gamma+memories[t].reward
        disc_rew[t]=run_add
    return disc_rew
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Memory=namedtuple('Memory',['obs','action','new_obs','reward','done'],rename=False)
    GAMMA=0.99
    LEARNING_RATE=0.002
    ENTROPY_BETA=0.01
    ENV_NAME='CartPole-v0'
    MAX_N_GAMES=5000
    n_games=0
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    now = datetime.datetime.now()
    datetime = "{}_{}.{}.{}".format(now.day,now.hour,now.minute,now.second)
    env = gym.make(ENV_NAME)
    obs = env.reset()
    writer = SummaryWriter(log_dir='content/runs/PG'+ENV_NAME+'_'+datetime)
    action_n = env.action_space.n
    agent_nn = PG_nn(env.observation_space.shape,action_n).to(device)
    optimezer = optim.Adam(agent_nn.parameters(),lr=LEARNING_RATE)

    experience = []
    tot_reward = 0
    n_iter = 0
    baseline = deque(maxlen=30000)
    game_rew = 0
    while n_games < MAX_N_GAMES:
        n_iter += 1
        act = agent_nn(torch.tensor(obs).to(device))
        #print(act)
        act_soft = F.softmax(act)
        #print(act_soft)
        # print(act_soft.detach.numpy())
        action = int(np.random.choice(np.arange(action_n),p=act_soft.cpu().detach().numpy(),size=1))
        #print(action)
        new_obs,reward,done,_ = env.step(action)

        game_rew += reward
        experience.append(Memory(obs=obs,action=action,new_obs=new_obs,reward=reward,done=done))
        obs = new_obs
        if done:
            disc_rewards = discounted_rewards(experience,GAMMA)

            baseline.extend(disc_rewards)
            disc_rewards -= np.mean(baseline)

            acts = agent_nn(torch.tensor([e.obs for e in experience]).to(device))
            game_act_log_softmax_t = F.log_softmax(acts,dim=1)[:,[e.action for e in experience]]
            #print(game_act_log_softmax_t)
            disc_rewards_t = torch.tensor(disc_rewards,dtype=torch.float32).to(device)

            l_entropy = ENTROPY_BETA*torch.mean(torch.sum(F.softmax(acts,dim=1)*F.log_softmax(acts,dim=1),dim=1))
            loss = -torch.mean(disc_rewards_t*game_act_log_softmax_t)
            loss = loss+l_entropy
            optimezer.zero_grad()
            loss.backward()
            optimezer.step()
            writer.add_scalar('loss',loss,n_iter)
            writer.add_scalar('reward',game_rew,n_iter)
            #print(n_games,loss.cpu().detach().numpy(),game_rew,np.mean(disc_rewards),np.mean(baseline))
            experience = []
            game_rew = 0
            obs = env.reset()
            n_games +=1
    writer.close()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
