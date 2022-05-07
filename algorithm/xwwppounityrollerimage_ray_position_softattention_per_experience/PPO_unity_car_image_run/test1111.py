import torch
import numpy as np

def array_tensor(array_numpy):
    pool =[]
    for i in array_numpy:
        i_tensor = torch.tensor(i)
        print("2222222222222",i_tensor.dtype)
        pool.append(i_tensor)
    return pool
def array_value(array_value):
    pool =[]
    for i in array_value:
        i_tensor = torch.tensor(i)
        i_numpy = i_tensor.numpy()
        pool.append(i_numpy[0])
    return pool
# state = [[1.2,2.5,3,4],
#          [4,5.1,6,7]]
#
# y = array_tensor(state)
# print("y",y)
state = [torch.tensor([-0.05]),torch.tensor([-0.05]),torch.tensor([-0.05]),torch.tensor([-0.05])]
y = array_list(state)
print("y11111111111111111111111",y)