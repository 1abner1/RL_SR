import time

import torch.nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import Tensor
import torch.nn.functional as F
dataset =Planetoid (root="data/planetoid",name = "Cora",transform= NormalizeFeatures()) #transform 预处理

print("dataset:", dataset)
data = dataset[0]
print("输入特征",dataset.num_features)
print("输出类别",dataset.num_classes)
print(data)
# x= 2708 篇论文，每篇论文有1433个向量，边的个数有10556 个边，

print("节点数量",data.num_nodes)
print("节点边",data.num_edges)

def visualize(h,color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.ion()
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:,0],z[:,1],s=70, c = color, cmap="Set2")
    plt.pause(0.9)
    plt.ioff()
    print("暂定1s")
    # plt.show()
    plt.close()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)

print("model",model)

# model.eval()
# out = model(data.x, data.edge_index)
# visualize(out,color=data.y)

optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out =model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim = 1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum())/int(data.test_mask.sum())
    return test_acc

for epoch in range(1,1000):
    loss = train()
    print("epoch:{} Loss:{}".format(epoch,loss))
    # print("训练结束")
    test_acc = test()
    print("准确率",test_acc)
    model.eval()
    out = model(data.x, data.edge_index)
    visualize(out,color=data.y)
    # time.sleep(0.1)

# model = GCN(hidden_channels =16)
# print("model",model)




