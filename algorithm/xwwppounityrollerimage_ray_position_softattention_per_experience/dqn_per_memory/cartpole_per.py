import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from prioritized_memory import Memory

EPISODES = 500

# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )

    def forward(self, x):
        return self.fc(x)


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and prioritized experience replay memory & target q network
class DQNAgent():
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.memory_size = 20000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 64
        self.train_start = 1000

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = DQN(state_size, action_size)
        self.model.apply(self.weights_init)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/cartpole_dqn')

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        target = self.model(Variable(torch.FloatTensor(state))).data
        old_val = target[0][action]
        target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])
        # 把这个经验存入到
        self.memory.add(error, (state, action, reward, next_state, done))
        #存入到sample中去，怎么没有找到村sample 这个池子呢

    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        mini_batch,idxs, is_weights = self.memory.sample(self.batch_size)   #只传入一个整数，为什么获取了min_batch 呢
        # print("mini_batch1111111111111111111111111111",mini_batch)
        # print("idxs2222222222222222222222222222", idxs)
        # print("is_weights333333333333333333333333", is_weights)
        # 对优先经验采样动作
        mini_batch = np.array(mini_batch).transpose()
        # print("mini_batch7777777777777777777777",mini_batch)

        states = np.vstack(mini_batch[0])    # 按理说状态应该从一堆状态中进行采样，看代码看是直接是随机数生成的
        # print("状态2222222222222222222",states)
        actions = list(mini_batch[1])
        # print("actions 2222222222222222222", actions)
        rewards = list(mini_batch[2])
        # print("rewards2222222222222222222", rewards)
        next_states = np.vstack(mini_batch[3])
        # print("next_states2222222222222222222", next_states)
        dones = mini_batch[4]
        # print("dones2222222222222222222", dones)

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float()
        pred = self.model(states)            #用数据-状态

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1) #用数据-动作

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)
        # print("one_hot_action222222222222222222", one_hot_action)  #64 组动作

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        # print("next_states444444444444444444444444",next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data     #目标的化使用的是下一时刻的动作值

        rewards = torch.FloatTensor(rewards)
        # print("rewards9999999999999999999",rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)
        # print("pred111111111111111111111",pred)
        # print("target111111111111111111111",target)

        errors = torch.abs(pred - target).data.numpy()   # 这里的erros 是50 组
        # print("eloor2222222222222222222222222222222222222",errors)

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            # print("idx111111111111111", idx)
            self.memory.update(idx, errors[i])
            # 更新经验池

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss.backward()

        # and train
        self.optimizer.step()


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)

    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        
        state = env.reset()
        # print("state22222222222222222222",state)
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)   # 将环境中获得的状态值传入到算法中
            # print("action777777777777777777777",action)
            next_state, reward, done, info = env.step(action)  #
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -10

            # save the sample <s, a, r, s'> to the replay memory   # 将这一排数据传入到经验池中
            agent.append_sample(state, action, reward, next_state, done)   #存数据
            # every time step do the training
            # print("agent.memory.tree.n_entries",agent.memory.tree.n_entries)
            if agent.memory.tree.n_entries >= agent.train_start:
                agent.train_model()  #是否应该返回一个状态值，取数据
                #优先经验池
                # print("开始训练")
            score += reward
            state = next_state
            # print("下时刻的状态00000000000000000",state)
            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 10
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig(r"D:\xwwppounityrollerimage_ray_position_softattention\dqn_per_memory\sace_graph\cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      agent.memory.tree.n_entries, "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model, r"D:\xwwppounityrollerimage_ray_position_softattention\dqn_per_memory\sace_graph")
                    sys.exit()
