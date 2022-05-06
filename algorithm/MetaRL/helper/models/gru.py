
import torch
import torch.nn as nn
from torch.distributions import Categorical
from helper.model_init import weight_init


class GRUActorCritic(nn.Module):
  def __init__(self, output_size, input_size, hidden_size=256):
    super(GRUActorCritic, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)  #torch直接使用了一个gru 中的参数，直接是一个
    self.relu1 = nn.ReLU()     #一个激活函数
    self.policy = nn.Linear(hidden_size, output_size)  #一个策略全连接层
    self.value = nn.Linear(hidden_size, 1)            # 一个值函数的值效果图
    self.apply(weight_init)                           # 权重初始化

  def forward(self, x, h):
    x, h = self.gru(x, h)
    x = self.relu1(x)
    val = self.value(x)
    dist = self.policy(x).squeeze(0)
    return Categorical(logits=dist), val, h

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([1, batchsize, self.hidden_size])