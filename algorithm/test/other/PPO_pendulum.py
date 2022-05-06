import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import os, time

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', default=True, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


class Agent():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 32

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

        self.writer = SummaryWriter('./tensorboardshow')

        if not os.path.exists('./param'):
            os.makedirs('./param')
        if not os.path.exists('./log'):
            os.makedirs('./log')
        if not os.path.exists('./tensorboardshow'):
            os.makedirs('./tensorboardshow')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
            #print("mu",mu)
            #print("sigma", sigma)
        dist = Normal(mu, sigma)
        action = dist.sample()   #动作值
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()
        # return action, action_log_prob

    def get_value(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def save_param(self):
        torch.save(self.anet.state_dict(), './param/ppo_anet_params.pkl')
        torch.save(self.cnet.state_dict(), './param/ppo_cnet_params.pkl')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        # start_buffer = self.counter % self.buffer_capacity == 0
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)

        print("下一时刻的状态", s_)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + args.gamma * self.cnet(s_)
            # print("目标crtic值的大小111111111111111", target_v)


        # target_v = r + args.gamma * self.cnet(s_)
        # target_v = target_v().no_grad()
        # print("目标crtic值的大小", target_v)

        adv = (target_v - self.cnet(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()
                # print("策略loss",action_loss)

                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])
                # print("值loss", value_loss)
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

                total_loss = action_loss + value_loss
                total_loss = total_loss.detach().numpy()
                # print("总loss",total_loss)

        del self.buffer[:]


def main():
    env = gym.make('Pendulum-v0')
    env.seed(args.seed)

    agent = Agent()

    training_records = []
    running_reward = -1000
    state = env.reset()
    for i_ep in range(100000):
        score = 0
        state = env.reset()

        for t in range(20):
            action, action_log_prob = agent.select_action(state)
            #print("action",type(action))
            #print("动作的概率值",action_log_prob)
            state_, reward, done, _ = env.step([action])
            if args.render:
                env.render()
            y = Transition(state, action, action_log_prob, (reward + 8) / 8, state_)
            # print("111111111111112222",y)
            # right = agent.store(y)
            print("right122222222222222222",agent.buffer)
            if agent.store(y):
                agent.update()
            score += reward
            state = state_

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        agent.writer.add_scalar('reward', running_reward, global_step=i_ep)

        if i_ep % args.log_interval == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))
        if running_reward > -15:
            print("Solved! Moving average score is now {}!".format(running_reward))
            agent.save_param()
            with open('log/ppo_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            break
    env.close()
    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("ppo.png")
    # plt.show()


if __name__ == '__main__':
    main()
