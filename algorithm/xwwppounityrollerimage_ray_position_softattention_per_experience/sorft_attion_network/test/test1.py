import torch
import torch.nn as nn
import torch.optim as optim

class actorcrtic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actorcrtic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    def actor_net(self):
        #actor 网络，演员网络输入状态维度，输出动作维度
        actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        print("actor",actor)
    def critic_net(self):
        # critic
        critic = nn.Sequential(
                        nn.Linear(self.state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        print("critic ",critic)

net = actorcrtic(state_dim =8,action_dim=2)
actor = net.actor_net()
crtic = net.critic_net()

loss_mese =torch.nn.MSELoss()
input = torch.tensor([0.06])
input = input.requires_grad_()
target = torch.tensor([0.05])
loss_value = loss_mese(input,target.float())
print("loss_value",loss_value)
optimizer_actor = optim.SGD(actor.parameters(), lr=0.0001)
optimizer_critic = optim.SGD(actor.parameters(), lr=0.0001)
optimizer_actor.zero_grad()   #清空过往梯度；
optimizer_critic.zero_grad()
loss_value.backward()    #反向传播，计算当前梯度；是否会传入到两个网络中
optimizer_actor.step()   #更新actor中w 参数
optimizer_critic.step()  #更新ctritic中w 参数
print("更新完成")