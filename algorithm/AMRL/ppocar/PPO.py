import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal  # 用于连续的动作
from torch.distributions import Categorical         # 用于离散的动作
import numpy as np
# import sys
# sys.path.append(r"D:\xwwppounityrollerimage_ray_position\sorft_attion_network")
import sorft_attion_network.soft_attention as MLP
from per_expericence import *
from per_expericence.prioritized_memory import Memory


################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')
# device = torch.device('cuda:0')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
    # print("Device set to : cpu")
print("Device set to : cpu")
# print("Device set to : Gpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


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
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        self.MLP = MLP.MLP()  #使用soft 注意力机制
        
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

        action = dist.sample()    #动作采用多元高斯分布采样
        action_logprob = dist.log_prob(action) #这个动作概率就相当于优先经验回放的is_weight
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)   #将状态输入到actor 网络中输出动作action
            
            action_var = self.action_var.expand_as(action_mean) #均值
            cov_mat = torch.diag_embed(action_var).to(device)  #方差
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)  #取动作的概率
        dist_entropy = dist.entropy()   #多元高斯分布的熵是什么，熵是两个分布的比值，熵是否理解为期望，所有值的概率*值，信息熵是一个值
        state_values = self.critic(state)    #将状态输入到critic 中得到的是状态值

        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory_size =2000
        #添加优先经验回放
        self.memory = Memory(self.memory_size)
        self.loss = np.random.rand(1)
        #添加优先经验回放
        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # print("执行初始化111111111111111111111111")
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                        {'params': self.policy.MLP.model.parameters(), 'lr': 0.05}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()  # mseloss = loss(x1,y1)=(xi-yi)^2
        # 这条代码xww 添加，为什么只用均方差，不使用交叉熵。
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        # self.loss = torch.tensor([])



    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

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

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # print("state222222222222222222222222222222222222222222222222222222shape:",state.shape)
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


    def update(self,state,action,reward,action_logsprob,done):
        self.states = state
        # print("传入的状态值3333333333444444444444444444444444444444444444444444444444444444444444444444444444444444444444",self.states)
        # print("传入的状态值33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333",self.buffer.states)
        # print("buffer状态111111111111111111112222222222222222222223333333333333333333333333333333333333333333333333333333", self.buffer.rewards)
        self.actions = action
        self.rewards = reward
        self.logprobs = action_logsprob
        self.is_terminals = done


        # print("输出的终端3333333333333333333333333333333333333333333",self.buffer.is_terminals)
        #原来的self.buffer.rewards 改为self.rewards

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # print("buffer状态111111111111111111112222222222222222222223333333333333333333333333333333333333",rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        t = self.buffer.states
        # t1 = np.shape(t)
        # print("\nstates1111111111111111111111111111:",t1)
        # t = torch.stack(self.buffer.states, dim=0)
        # print("t2222222222222222222222222222222222222222222222222222222222222222:",t[0])
        # print("t3333333333333333333333333333333333333333333333333333333333333333:",t[1])
        # print("t4444444444444444444444444444444444444444444444444444444444444444:",t[2])
        # print("t5555555555555555555555555555555555555555555555555555555555555555:",t[3])
        # t2 = torch.stack(t)
        # print("状态组",t2)
        # print("buffer状态11111111111111111111222222222222222222222",self.buffer.states)
        # print("buffer状态11111111111111111111222222222222222222222", self.buffer.actions)
        # print("buffer状态11111111111111111111222222222222222222222", self.buffer.logprobs)
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # print("旧的状态值11111111111111112222222222222223333",old_states)
            # print("旧的动作4444444444444444444444444444444444444", old_states)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            # print("surr1",surr1)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # print("surr2", surr2)

            # final loss of clipped objective PPO
            # actor_loss = -torch.min(surr1, surr2).mean()
            # critic_loss = 0.5*self.MseLoss(state_values, rewards)
            # 这个更新方式为clip 的ppo 而不是pential 的ppo (使用kl散度)，目标是让获得评估值和奖励函数进行对比，state_value是通过critic输出的，actor获得动作
            loss2 = abs(state_values-rewards).detach()
            # print("loss2444444444444444444444444444444444444444444444",loss2)
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy   # dist_entropy 为多元高斯分布的信息熵(传入动作的信息熵) 最后一项表示的增加探索项。
            # print("state_values11111111111111111111111111111111111",state_values)
            # kl 散度表示为=交叉熵-熵；MseLoss均方损失函数，CrossEntropyLoss 交叉熵；MES 是神经网络的nn.loss,(yi-xi)^2 - 多元高斯分布的熵（两个熵相减等于crtic 的loss）
            #self.CrossEntropyLoss()
            # self.loss = loss
            # 加入软注意力之后，需要单独再加一个软注意力机制的loss,软注意力的loss 更新我应该怎么更新呢？
            self.loss = loss
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            loss1 = loss.mean()
            # print("ppoloss222222222222",loss.mean())
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return loss1,loss2
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


    def Store_Sample(self,state, action, reward, next_state, done):
        # error = self.loss   #需要传入loss作为值
        error = np.random.rand(1)
        # print("error111111111111111111111111111",error)
        self.memory.add(error, (state, action, reward, next_state, done))
        
        
       


