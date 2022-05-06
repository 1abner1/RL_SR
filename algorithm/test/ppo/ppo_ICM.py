import os
from datetime import datetime
import numpy as np
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import time

# set device to cpu or cuda
device = torch.device('cpu')
# if(torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.Tanh(),
                nn.Linear(100, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 100),
                nn.Tanh(),
                nn.Linear(100,100),
                nn.Tanh(),
                nn.Linear(100, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO():
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.writer = SummaryWriter('./log9')

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            policyloss = -torch.min(surr1, surr2)
            policyloss = policyloss.mean().cpu().detach().numpy()
            valueloss =  self.MseLoss(state_values, rewards)
            valueloss = valueloss.mean().cpu().detach().numpy()
            entroyloss = dist_entropy
            entroyloss = entroyloss.mean().cpu().detach().numpy()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            total_loss = loss.mean().cpu().detach().numpy()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

        return total_loss,policyloss,valueloss,entroyloss
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def train():
    env_name = "Pendulum-v0"      #mountaincar-v0 Pendulum-v0 CartPole-v0
    has_continuous_action_space = True  # continuous action space; else discrete
    max_ep_len = 200                   # max timesteps in one episode
    action_std = 0.5                   # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    K_epochs = 64               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.9            # discount factor
    lr_actor = 0.004       # learning rate for actor network
    lr_critic = 0.003       # learning rate for critic network
    random_seed = 1         # set random seed if required (0 = no random seed)
    max_episode = 200000000

#---------------------开始训练--------------------------------
    env = gym.make(env_name)
    # state space dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    directory = "model"
    if not os.path.exists(directory):
          os.makedirs(directory)
    # directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)
    # checkpoint_path = directory + '/' + "PPO_{}_{}.pth".format(env_name, random_seed)
    # checkpoint_path = "./model/ppo_model.pth"
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print_running_reward = 0
    time_step = 0
    render = True
    # training loop
    # while time_step <= max_training_timesteps:
    for episode in range(1, max_episode):
        state = env.reset()
        current_ep_reward = 0
        for step in range(1, max_ep_len+1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward
            if render:
                env.render()
            # update PPO agent
            total_loss1 = 0
            if step % max_ep_len == 0:
                total_loss1,policy_loss,value_loss,entropy_loss = ppo_agent.update()
            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if step % max_ep_len == 0:
                # print average reward till last episode
                print_avg_reward = current_ep_reward / max_ep_len
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode:{}\tTimestep:{}\tAverage Reward:{}\ttotal_loss:{:.3f}".format(episode, time_step, print_avg_reward,total_loss1))
                # 保存tensorboard 显示数据
                ppo_agent.writer.add_scalar('reward/reward', print_avg_reward, episode)
                ppo_agent.writer.add_scalar('loss/entropy_loss', entropy_loss, episode)
                ppo_agent.writer.add_scalar('loss/total_loss', total_loss1, episode)
                ppo_agent.writer.add_scalar('loss/policy_loss', policy_loss, episode)
                ppo_agent.writer.add_scalar('loss/value_loss', value_loss, episode)
                ppo_agent.writer.flush()
                ppo_agent.writer.close()
                print_running_reward = 0
            # save model weights
            # print("输出步数",step)
            if step == max_ep_len:
                checkpoint_path = "./model/ppo_model.pth"
                ppo_agent.save(checkpoint_path)
                print("model saved")
            # break; if the episode is over
            if done:
                break
        print_running_reward += current_ep_reward

    env.close()
#---------------------结束训练--------------------------------

def test():
    env_name = "Pendulum-v0"
    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving
    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames
    total_test_episodes = 10    # total num of testing episodes
    K_epochs = 100               # update policy for K epochs
    eps_clip = 0.3              # clip parameter for PPO
    gamma = 0.96                # discount factor
    lr_actor = 0.0007           # learning rate for actor
    lr_critic = 0.003           # learning rate for critic
#-------------------------------开始测试------------------------------------
    env = gym.make(env_name)
    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    directory = "model"
    checkpoint_path = "./model/ppo_model.pth"
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    test_running_reward = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                time.sleep(frame_delay)
            if done:
                break
        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))


if __name__ == '__main__':
    train()
    # test()
    
    
    
    
    
    
    
    
