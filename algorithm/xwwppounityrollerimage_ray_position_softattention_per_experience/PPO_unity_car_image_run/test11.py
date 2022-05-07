# import pandas as pd
#
# # def save_final_episode(episdode1):
# #     os.makedirs(os.path.join('..', 'episode_step'), exist_ok=True)
# #     episode_step = os.path.join('..', 'episode_step', 'episode.csv')
# #     with open(episode_step, 'w', encoding='utf-8') as f:
# #         f.write(str(episdode1))  # 列名
# #     return episode_step
# #
# # def load_final_episode():
# #     with open(r"D:\xwwppounityrollerimage\episode_step\1.txt", 'r') as f:
# #         data = f.read()
# #     return int(data)
# #
# # t = load_final_episode()
# # print("t1111111111111111",t)
# # import torch
# # from torch.distributions import MultivariateNormal
# # x1 = torch.zeros(2)  #mu  类似与均值
# # x2 = torch.eye(2)    #sigama 类似等于方差
# # print("x1",x1)
# # print("x1",x2)
# # t = MultivariateNormal(x1,x2)  # 得到一个联合概率密度函数
# # print("t",t)
# # 第三次测试
# import torch
# import torch.nn as nn
# from numpy import *
#
#
# # Args:
# #         in_channels (int): Number of channels in the input image  {输入通道数}
# #         out_channels (int): Number of channels produced by the convolution  {输出通道}
# #         kernel_size (int or tuple): Size of the convolving kernel
# #         stride (int or tuple, optional): Stride of the convolution. Default: 1
# #         padding (int or tuple, optional): Zero-padding added to both sides of
# #             the input. Default: 0
# #         padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
# #             ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
# #         dilation (int or tuple, optional): Spacing between kernel
# #             elements. Default: 1
# #         groups (int, optional): Number of blocked connections from input
# #             channels to output channels. Default: 1
# #         bias (bool, optional): If ``True``, adds a learnable bias to the
# #             output. Default: ``True``
# #
#
# conv1 = nn.Conv1d(3, 2, 2, 1)  # in, out, k_size, stride
#
# a = torch.ones(1, 5, 3)  # b_size, n, 3  # 点云格式
# print("a11111111",a)
# a = a.permute(0, 2, 1)
# # a = torch.ones(3)
# print('a:\n', a)
#
# b = conv1(a)
# print('b:\n', b)
# print(b.shape)  # b_size, ch, length
#
#
# 第四次卷积
import torch
import torch.nn as nn
from torch.autograd import Variable

# input = torch.randn(404)
# lay = nn.Linear(404,8)
# OUT= lay(input)
# print("output",OUT.backward())
# # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
# # input = input.permute(0,1)
# # input = Variable(input)
# # out = conv1(input)
# # print(out.size())
x =[1,2,3]
x1 =[5,6,7]
y =x.extend(x1)
print(y)

