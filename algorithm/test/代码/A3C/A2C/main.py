# This is a sample Python script.
import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import namedtuple
from collections import  deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class A2C_nn(nn.Module):
    def __init__(self,input_shape,n_actions):
        super(A2C_nn, self).__init__()
        self.lp = nn.Sequential(
            nn.Linear(input_shape[0],64),
            nn.ReLU()
        )
        self.policy = nn.Linear(64,n_actions)
        self.value = nn.Linear(64,1)
    def forward(self,x):
        l = self.lp(x.float())
        return self.policy(l),self.value(l)

def calculate_loss(memories,nn,writer):
    rewards = torch.tensor(np.array([m.reward for m in memories],dtype=np.float32))
    log_val = nn(torch.tensor(np.array([m.obs for m in memories],dtype=np.float32)))
    act_log_softmax = F.log_softmax(log_val[0],dim=1)[:,np.array([m.action for m in memories])]
    # Advantage
    adv = (rewards - log_val[1].detach())
    # Actor loss
    pg_loss = -torch.mean(act_log_softmax*adv)
    # critic loss
    vl_loss = F.mse_loss(log_val[1].squeeze(-1),rewards)
    # entropy loss
    entropy_loss = ENTROPY_BETA*torch.mean(torch.sum(F.softmax(log_val[0],dim=1)*F.log_softmax(log_val[0],dim=1),dim=1))

    # total loss = policy loss+ critic loss - entropy loss
    loss = pg_loss + vl_loss - entropy_loss

    # add scalar to the writer
    writer.add_scalar('loss',float(loss),n_iter)
    writer.add_scalar('pg_loss',float(pg_loss),n_iter)
    writer.add_scalar('vl_loss', float(vl_loss), n_iter)
    writer.add_scalar('entropy_loss', float(entropy_loss), n_iter)
    writer.add_scalar('actions', np.mean([m.action for m in memories]), n_iter)
    writer.add_scalar('adv', float(torch.mean(adv)), n_iter)
    writer.add_scalar('act_lgsoft', float(torch.mean(act_log_softmax)), n_iter)
    return loss

class Env:
    game_rew = 0
    last_game_rew = 0
    def __init__(self,env_name,n_steps,gamma):
        super(Env, self).__init__()

        self.env = gym.make(env_name)
        self.obs = self.env.reset()

        self.n_steps = n_steps
        self.action_n = self.env.action_space.n
        self.observation_n = self.env.observation_space.shape[0]
        self.gamma = gamma
    def step(self,agent):
        memories = []
        for s in range(self.n_steps):
            pol_val = agent(torch.tensor(self.obs))
            s_act = F.softmax(pol_val[0])
            action = int(np.random.choice(np.arange(self.action_n),p=s_act.detach().numpy(),size=1))
            new_obs,reward,done,_=self.env.step(action)

            memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=reward, done=done))
            self.game_rew += reward
            self.obs = new_obs
            if done:
                self.done = True
                self.run_add = 0
                self.obs = self.env.reset()
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                break
            else:
                self.done = False
        if not self.done:
            self.run_add = float(agent(torch.tensor(self.obs))[1])
        return self.discounted_rewards(memories)
    def discounted_rewards(self,memories):
        upd_memories = []
        for t in reversed(range(len(memories))):
            if memories[t].done:
                self.run_add = 0
            self.run_add = self.run_add*self.gamma + memories[t].reward
            upd_memories.append(Memory(obs=memories[t].obs,action=memories[t].action,new_obs=memories[t].new_obs,reward=self.run_add, done=memories[t].done))
        return upd_memories[::-1]

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    Memory = namedtuple('Memory',['obs','action','new_obs','reward','done'],rename=False)
    GAMMA = 0.95
    LEARNING_RATE = 0.003
    ENTROPY_BETA = 0.01
    ENV_NAME = 'CartPole-v0'
    MAX_ITER = 100000
    N_ENVS = 40

    CLIP_GRAD = 0.1
    device= 'cpu'
    now = datetime.datetime.now()
    datetime = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    envs = [Env(ENV_NAME,1,GAMMA) for _ in range(N_ENVS)]
    writer = SummaryWriter(log_dir='content/runs/A2C'+ENV_NAME+'_'+datetime)
    Env = gym.make(ENV_NAME)
    agent_nn = A2C_nn(Env.observation_space.shape,Env.action_space.n).to(device)
    optimizer = optim.Adam(agent_nn.parameters(),lr = LEARNING_RATE,eps=1e-3)
    experience = []

    n_iter = 0
    while n_iter < MAX_ITER:
        n_iter += 1
        memories = [mem for env in envs for mem in env.step(agent_nn)]
        losses = calculate_loss(memories,agent_nn,writer)
        optimizer.zero_grad()
        losses.backward()
        clip_grad_norm(agent_nn.parameters(),CLIP_GRAD)
        optimizer.step()

        writer.add_scalar('rew', np.mean([env.last_game_rew for env in envs]), n_iter)
        print(n_iter, np.round(float(losses), 2), 'rew:', np.round(np.mean([env.last_game_rew for env in envs]), 2))

    writer.close()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
