import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple
from torch.nn import init
import matplotlib.pyplot as plt

data = namedtuple('Data', ['s', 'a', 'a_log_p', 'r', 's_'])


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))  # 激活函数
        return (mu, sigma)


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 1
        self.action_net = nn.Sequential(
            nn.Linear(6, 3),  # 输入当前状态和下一时刻状态，输出预测动作a
            # nn.ReLU(),
            nn.Linear(3,feature_output)
        )
        self.state_net = nn.Sequential(
            nn.Linear(4, 512),
            nn.Linear(512, output_size)   # 输入为当前状态和当前的动作，预测出下一时刻的状态为3维
        )

    def forward(self, state, next_state, action):
        # state, next_state, action = inputs
        state = state.to(torch.float32)
        next_state = next_state.to(torch.float32)
        co_state = state.append(next_state)
        predi_action = self.action_net(co_state)
        # print("输出当前状态",predi_action)
        # # print("输出下一时刻状态", encode_next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 0)
        print("输出预测的动作",pred_action)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action


class PPO():

    def __init__(self):
        self.training_step = 0
        self.anet = Actor().float()
        self.cnet = Critic().float()
        self.buffer = []
        self.counter = 0
        self.max_epoch = 100
        self.gamma = 0.9
        self.buffer_capacity = 2  # 输入到村buffer的长度
        self.batch_size = 8
        self.total_loss = []
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

        self.writer = SummaryWriter('./log')

        if not os.path.exists('./model'):
            os.makedirs('./model')  # model 用来存放".pth文件"
        if not os.path.exists('./log'):
            os.makedirs('./log')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)  # 将状态输入到actor网络中，输出正态分布的mu值和sigama值，相当于输入一个状态输出为状态的均值和方差和值
        dist = Normal(mu, sigma)  # 关于动作的正态分布，
        action = dist.sample()  # 在动作正太分布中随机采取动作。
        action_log_prob = dist.log_prob(action)  # 输出动作的概率值
        action = action.clamp(-2.0, 2.0)  # 通过正太分布的参数来限定动作的范围
        return action.item(), action_log_prob.item()  # 遍历取出所有动作和动作的概率

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)  # 不确定几行，但可以确定是一列
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)
        # print("下一时刻的状态",s_)

        old_action_prob = torch.tensor([t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        # print("奖励函数", r.std())
        r = (r - r.mean()) / (r.std() + 1e-5)  # 奖励函数是一个tensor格式 ？

        with torch.no_grad():
            # print("价值",self.cnet(s_))
            target_v = r + self.gamma * self.cnet(s_)  # 目标crtic值 用来将目标值的tensor 输出没有tensor 类型
            # print("target_v3333333333333333333",target_v)
        current_v = self.cnet(s)
        adv = (target_v - current_v).detach()  # 使用优势函数切断优势函数的反向传播,还是一个值

        for _ in range(self.max_epoch):  # self
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu, sigma)
                action_log_prob = dist.log_prob(a[index])
                ratio = torch.exp(action_log_prob - old_action_prob[index])  # 学习率

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

                self.total_loss = action_loss + value_loss
                self.total_loss = self.total_loss.detach().numpy()

                # print("总loss", self.total_loss)

            del self.buffer[:]

        return self.total_loss

    def save_buffer(self, data):
        self.buffer.append(data)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def save_model(self):
        torch.save(self.anet.state_dict(), './model/ppo_anet_params.pth')
        torch.save(self.cnet.state_dict(), './model/ppo_cnet_params.pth')


def main():
    seed = 2
    env = gym.make('Pendulum-v0')
    env.seed(seed)
    agent = PPO()
    agentICM = ICMModel(input_size=3,output_size=1)   #input_size 状态的维度，输出为动作的维度
    print("icmagenticm",agentICM)
    max_step = 20
    sample_size = 10
    max_epoch = 1000
    render = True
    for epoch in range(max_epoch):
        total_reward = 0
        state = env.reset()
        # buffer_s,buffer_a,
        for step in range(max_step):
            action, action_prob = agent.select_action(state)
            state_, reward, done, _ = env.step([action])
            # print("每一步的奖励值",reward)
            if render:
                env.render()
            data_group = data(state, action, action_prob, reward, state_)
            agent.save_buffer(data_group)
            # print("agent", agent.buffer)
            if agent.save_buffer(data_group):
                total_loss = agent.update()
                # print("输出状态", torch.from_numpy(state))
                real_next_state_feature, pred_next_state_feature, pred_action = agentICM.forward(torch.tensor(state), torch.from_numpy(state_), torch.tensor(action))
                print("输出icm之后的结果",real_next_state_feature,pred_next_state_feature,pred_action)
                # agent.save_model()
                # print("总loss",total_loss)
            total_reward += reward
            state = state_
        mean_reward = total_reward / max_step if max_step > 0 else 0.0
        print('Train_epoch:{}\taverage_reward:{:.2f}\t total_loss:{:.3f}'.format(epoch, mean_reward, total_loss))
        agent.writer.add_scalar('reward', mean_reward, epoch)
        agent.writer.add_scalar('total_loss', total_loss, epoch)

    # plt.plot(epoch, mean_reward)
    # plt.title('PPO')
    # plt.xlabel('epoch')
    # plt.ylabel('averaged_reward')
    # plt.savefig("ppo.png")
    # plt.show()
    env.close()


if __name__ == '__main__':
    main()
