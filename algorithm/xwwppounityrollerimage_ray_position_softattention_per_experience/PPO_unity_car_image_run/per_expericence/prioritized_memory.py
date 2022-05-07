import random
import numpy as np
import sys

import torch

sys.path.append(r'D:\xwwppounityrollerimage_ray_position_softattention_per_experience\PPO_unity_car_image_run\per_expericence')
from SumTree import SumTree

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        x = error
        # x = np.absolute(x)
        # x = torch.Tensor(x)
        # print("x",type(x))
        x = abs(x)
        x1 = self.e
        x2 = self.a
        y =(x+x1)**x2
        # return (np.abs(error) + self.e) ** self.a
        return y
    def add(self, error, data):
        # print("error11111111111111111111111111111111111111111111111111111111111",error)
        max_p1 = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p1 == 0:
            max_p1 = 1.0
        p = self._get_priority(error)   #由于loss 的值给的是一个随机数，所以其并没有变化 #error 没有用上
        # print("p的值为222222222222222",p)
        self.tree.add(max_p1, data)  # p是重要性，data是经验

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)   #决策树
            # print("data111111111111111111",data)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        # print("xx1",self.tree.n_entries * sampling_probabilities)
        # print("y1111", -self.beta)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        # is_weight = 1/self.tree.n_entries * sampling_probabilities
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
