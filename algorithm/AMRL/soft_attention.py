import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt
import torch
import torch.nn as nn

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



class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

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






