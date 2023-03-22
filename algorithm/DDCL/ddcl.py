#基于双重决策和先验知识的虚实迁移持续学习方法
# 首先创建代码总体框架
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
#感知阶段
class Feature_Gain()
     def SR_Fusion(self):
        pass
     def Graph_Bulid(self):
        pass
    def Gan(self):
        pass

#决策阶段
class Policy_Learning(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Policy_Learning, self).__init__()

    def actor(self,state_dim,action_dim):
        nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
        )
    def critic(self,state_dim):
         nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def update(self):
        pass
    def Get_action(self,state):
        action_mean = actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()  # 动作采用多元高斯分布采样
        action_logprob = dist.log_prob(action)  # 这个动作概率就相当于优先经验回放的is_weight

    return action.detach(), action_logprob.detach()
class Rule_Control():
      pass

class Discriminator_1():
      pass


class Discriminator_2():
    pass

#执行阶段
class Execution_Phase()
    def main(self):
        pass