import sys,os

from torch.distributions import Categorical
import torch.nn.functional as F
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np
import torch
import torch.optim as optim
import datetime
from multiprocessing_env import SubprocVecEnv
from models import TwoHeadNetwork
class A2CConfig:
    def __init__(self):
        self.algo = 'A3C'
        self.env_name = 'CartPole-v0'
        self.n_envs = os.cpu_count()
        self.gamma = 0.99
        self.hidden_dim = 256
        self.lr = 1e-4
        self.max_frames = 30000
        self.n_steps = 5
        self.device = "cpu"
def make_envs(env_name):
    def _thunk():
        env = gym.make(env_name)
        env.seed(2)
        return env
    return _thunk
def test_env(env,model,vis=False):
    state = env.reset()
    if vis:env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        dist = F.softmax(dist,dim=1)
        dist = Categorical(dist)
        next_state,reward,done,_ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis:env.render()
        total_reward += reward
    return total_reward
def compute_returns(next_value,rewards,masks,gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma*R*masks[step]
        returns.insert(0,R)
    return returns

def train(cfg,envs,writer):
    env = gym.make(cfg.env_name)
    env.seed(10)
    state_dim = envs.observation_space.shape[0]
    action_dim = env.action_space.n
    model = TwoHeadNetwork(state_dim,action_dim)
    optimizer = optim.Adam(model.parameters())
    frame_idx = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()
    testepisode = 0
    finish = False
    start_time = datetime.datetime.now()
    while frame_idx < cfg.max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0


        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist,value = model(state)
            dist = F.softmax(dist,dim=1)
            dist = Categorical(dist)
            action = dist.sample()
            next_state,reward,done,_ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            frame_idx += 1
            if frame_idx %100 ==0:
                test_reward = np.mean([test_env(env,model) for _ in range(100)])
                print(f"frame_idx:{frame_idx},test_reward:{test_reward}")
                test_rewards.append(test_reward)
                testepisode += 1
                print(int((datetime.datetime.now() - start_time).total_seconds()))
                writer.add_scalar("averageR",float(test_reward),testepisode)
                writer.add_scalar("time_reward",float(test_reward),(datetime.datetime.now() - start_time).total_seconds())
                if test_ma_rewards:
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward)
                if test_reward >=195.0:
                    finish = True
                    break


        if finish:
            break
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value,rewards,masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns-values
        actor_loss = -(log_probs*advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss+0.5*critic_loss-0.001*entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return test_rewards,test_ma_rewards
if __name__ == '__main__':
    cfg = A2CConfig()
    now = datetime.datetime.now()
    datetime1 = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    writer = SummaryWriter(log_dir='runs/A3CCartPole-v0' + '_' + datetime1)
    envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = SubprocVecEnv(envs)

    rewards,ma_rewards = train(cfg,envs,writer)
    writer.close()
