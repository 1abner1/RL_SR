import torch
import numpy as np
# loss=torch.nn.MSELoss()
# w=np.array([1.0,2.0,3.0])
# w1=np.array([1.0,2.0,2.0])
# loss1 = loss(torch.tensor(w),torch.tensor(w1))
# print("loss平均值",loss1)
# t2 = loss.backward(retain_graph=True)
# print("t22222222222222222222222",t2)
loss = torch.nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)  # 输入必须要求有梯度
print("输入为33333333333333333",input)
target = torch.randn(3, 5)
output = loss(input, target)
print("输出loss",output)
output.backward()
