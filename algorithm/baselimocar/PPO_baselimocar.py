
from collections import namedtuple
from itertools import count
import os, time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from unity_API import UnityWrapper

# Parameters
gamma = 0.99
render = True
seed = 1
log_interval = 10

Unity_env = UnityWrapper(train_mode=True, base_port=5004)

env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('./curve_log')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('./model'):
            os.makedirs('./model')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_model(self,model_actor_pth,model_critc_pth):
        torch.save(self.actor_net.state_dict(), model_actor_pth)
        torch.save(self.critic_net.state_dict(), model_critc_pth)

    def load_model(self,model_actor_pth,model_critc_pth):
        self.actor_net.load_state_dict(torch.load(model_actor_pth,map_location=lambda storage, loc: storage))
        self.critic_net.load_state_dict(torch.load(model_critc_pth,map_location=lambda storage, loc: storage))


    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        Reward1 = 0
        Gt = []
        for r in reward[::-1]:
            Reward1 = r + gamma * Reward1
            Gt.insert(0, Reward1)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    self.writer.add_scalar("reward",round(Reward1,2),i_ep)
                    print('I_ep {} , train_step {} ,total_reward {}'.format(i_ep,self.training_step,round(Reward1,2)))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                Value = self.critic_net(state[index])
                delta = Gt_index - Value
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, Value)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

        return

    
def train():
    agent = PPO()
    max_step = 300
    model_actor_param_pth = './model/actor.pth'
    model_critic_param_pth = './model/critic.pth'

    for i_epoch in range(1000):
        state = env.reset()
        # if render: env.render()
        for t in range(max_step):
            # print("t",t)
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state

            # done 的条件：杆子角度太大，离开直线
            if done:
                # print("完成了一个回合：",done,i_epoch)
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
                agent.save_model(model_actor_param_pth,model_critic_param_pth)
                break

def test():
    agent = PPO()
    max_step = 300
    model_actor_param_pth = './model/actor.pth'
    model_critic_param_pth = './model/critic.pth'
    agent.load_model(model_actor_param_pth,model_critic_param_pth)

    for i_epoch in range(1000):
        state = env.reset()
        for t in range(max_step):
            # print("t",t)
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state
            if done:
                break


if __name__ == '__main__':
    # train()
    test()
    print("end")
