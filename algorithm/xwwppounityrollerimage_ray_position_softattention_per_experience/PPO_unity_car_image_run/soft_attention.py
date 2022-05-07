import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):#自定义类 继承nn.Module

    def __init__(self):#初始化函数
        super(MLP, self).__init__()#继承父类初始化函数

        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(inplace=True),
        )#自定义实例属性 model 传入自定义模型的内部构造 返回类

    def forward(self, x):
        x = self.model(x)
        #x传入自定义的model类 返回经过模型后的输出
        return x

# # 输入参数
# s1 =np.random.randn(1,784)
# s2 =np.random.randn(1,784)
# s3 =np.random.randn(1,784)
# net = MLP()
# print("net",net)
# #input1 图像
# tensor_s1 = torch.tensor(s1)
# tensor_s1 = tensor_s1.to(torch.float32)
# output_w1 =net(tensor_s1[0])
# #input1 图像
#
# #input2 雷达
# tensor_s2 = torch.tensor(s2)
# tensor_s2 = tensor_s2.to(torch.float32)
# output_w2 =net(tensor_s2[0])
# #input2 雷达
#
# #input2 位置
# tensor_s3 = torch.tensor(s3)
# tensor_s3 = tensor_s3.to(torch.float32)
# output_w3 =net(tensor_s3[0])
# #input2 位置
#
# optimizer = optim.SGD(net.parameters(), lr=0.0001)
# target = torch.tensor([0.05])
#
# loss = torch.nn.MSELoss()   # 需要先定义使用什么损失函数
# out_loss =loss(output_w1, target.float())
# print("loss",out_loss.mean())
# optimizer.zero_grad()   #清空过往梯度；
# out_loss.backward()  #反向传播，计算当前梯度；
# optimizer.step()     #更新w 参数






