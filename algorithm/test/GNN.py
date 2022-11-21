import numpy as np

A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
], dtype=float)

X = np.matrix([
    [i, -i]
    for i in range(A.shape[0])
], dtype=float)

# 增加自环
I = np.matrix(np.eye(A.shape[0]))

A_hat = A + I
print(A_hat * X)

# 对特征表征进行归一化处理
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))
print(D_hat)

# 添加权重
W = np.matrix([
    [1, -1],
    [-1, 1]
])
print(D_hat ** -1 * A_hat * X * W)
# 减小输出特征表征的维度，我们可以减小权重矩阵 W 的规模
W = np.matrix([
    [1],
    [-1]
])
print(D_hat ** -1 * A_hat * X * W)


# 添加激活函数
def relu(x):
    return np.maximum(0, x)


print(relu(D_hat ** -1 * A_hat * X * W))

print("===============================")
'''构建GCN'''
from networkx import to_numpy_matrix, karate_club_graph

zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))


def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat ** -1 * A_hat * X * W)


H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}

print(output)
import matplotlib.pyplot as plt

plt.plot(output[:, 0], output[:, 1], '*')
plt.show()