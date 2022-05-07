import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class actorcrtic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actorcrtic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        #actor 网络，演员网络输入状态维度，输出动作维度
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        print("actor",self.actor)
        # critic  输入状态输入value
        self.critic = nn.Sequential(
                        nn.Linear(self.state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        print("critic ",self.critic)
    def forward(self, x):          #示例化之后自动调用forward这个函数 x是什么？一维向量?
        actor = self.actor(x)
        critic = self.critic(x)
        return actor,critic

net = actorcrtic(state_dim =8,action_dim=2)
# actor = net.actor_net()
# crtic = net.critic_net()
x1 = np.random.randn(8)
x1 =torch.tensor(x1)
x1 = x1.float()
print("x1",x1)
actor1, critic1 = net(x1)     #应该根据actor1和critic1 作为输入和目标值，输入值
print("actor1",actor1)
print("critic1",critic1)
loss_mese =torch.nn.MSELoss()
input = torch.tensor([0.06])
input = input.requires_grad_()
target = torch.tensor([0.05])
loss_value = loss_mese(input,target.float())
print("loss_value",loss_value)
# 随机梯度下降法
print("输入到优化器中的参数net.parameters",net.parameters())
# for _,param in enumerate(net.named_parameters()):
#     print(param[0])
#     print(param[1])
#     print('----------------')
optimizer_actor = optim.SGD(net.parameters(), lr=0.0001)   #net.parameters中的参数是包含两个网络中的参数吗？把参数传入到优化器中
# optimizer_actor = optim.SGD(actor.parameters(), lr=0.0001)
# optimizer_critic = optim.SGD(actor.parameters(), lr=0.0001)
# optimizer_actor.zero_grad()   #清空过往梯度；
# optimizer_critic.zero_grad()
loss_value.backward()    #反向传播，计算当前梯度；是否会传入到两个网络中
# optimizer_actor.step()   #更新actor中w 参数
# optimizer_critic.step()  #更新ctritic中w 参数
print("更新完成")