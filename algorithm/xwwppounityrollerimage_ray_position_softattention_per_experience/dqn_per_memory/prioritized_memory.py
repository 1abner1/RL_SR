import random
import numpy as np
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
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, data):
        p = self._get_priority(error)
        self.tree.add(p, data)  # p是重要性，data是经验

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
