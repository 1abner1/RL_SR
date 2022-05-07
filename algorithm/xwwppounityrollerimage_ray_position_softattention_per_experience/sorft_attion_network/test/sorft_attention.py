import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

s1 =np.random.randn(1,784)
s2 =np.random.randn(1,8)
s3 =np.random.randn(1,8)
s4 =np.random.randn(1,8)

class MLP(nn.Module):#自定义类 继承nn.Module

    def __init__(self):#初始化函数
        super(MLP, self).__init__()#继承父类初始化函数

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 1),
            nn.ReLU(inplace=True),
        )#自定义实例属性 model 传入自定义模型的内部构造 返回类

    def forward(self, x):
        x = self.model(x)
        #x传入自定义的model类 返回经过模型后的输出
        return x

net = MLP()
print("net",net)
tensor_s1 = torch.tensor(s1)
tensor_s11 = tensor_s1.to(torch.float32)
output_w1 = net(tensor_s11[0])
# print("输出的参数为22222222222222222222",output_w1)
# for _,param in enumerate(net.parameters()):
#     # print(param)
#     print('----------------')
optimizer = optim.SGD(net.parameters(), lr=0.0001)
target = torch.tensor([0.05])

# output_w1 =output_w1.unsqueeze(dim=0)
# output_w1 = output_w1.detach()
# output_w1 = output_w1.requires_grad_()
# loss = torch.nn.MSELoss(output_w1[0], target.long()) #input 输入是二维数据，目标是long类型
loss = torch.nn.MSELoss()
out_loss =loss(output_w1, target.float())
print("loss",out_loss.mean())
optimizer.zero_grad()   #清空过往梯度；
out_loss.backward()  #反向传播，计算当前梯度；
optimizer.step()     #更新w 参数






