import torch.nn as nn
import torch.nn.functional as F

import torch

# 如果有gpu的可使用GPU加速
# device = torch.device("cpu")
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

transformer = transforms.Compose([transforms.Resize((84, 84)),    #resize 调整图片大小
                                  # transforms.RandomHorizontalFlip(), # 水平反转
                                  transforms.ToTensor(),  # 0-255 to 0-1 归一化处理
                                  # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  #归一化
                                  ])
class CNNNet(nn.Module):
    # 150X150
    # (width-kernel_size + 2padding)/sride +1
    # (150-3+2*1)/1 +1
    def __init__(self):
        super(CNNNet, self).__init__()
        # 定义第一个卷积层 84 *84 *3
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1,padding=1,bias=False)
=======
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1,padding=1)
>>>>>>> master
        # 卷积之后的图像大小out_channels*(84-kernel_size=2+2*padding=1)/stride=1 + 1 ;(84-2 + 2*1)
        # 定义第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        # (85 -2 +2*1)/1 + 1=86  输出通道数 16*86*86
        # 定义第二个卷积层
<<<<<<< HEAD
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1,padding=1,bias=False)
=======
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1,padding=1)
>>>>>>> master
        #(86-2+2*1)/1  +1 ;3*83 *83
        # 定义第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        #16*88*88
        # 定义第一个全连接层
        self.fc1 = nn.Linear(16*84*84, 4096)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(4096, 8)
        # n_channels(int)：输入图片的通道数目，彩色图片的通道数为3(RGB)
        # out_channels(int): 卷积产生的通道数
        # kernel_size(int or tuple)：卷积核的尺寸，单个值则认为卷积核长宽相同
        # stride(int or tuple)：卷积步长
        # padding(int or tuple, optional)：输入的每一条边填充0的圈数，参数可选，默认为0
        # bias(bool, optional)：如果bias=True，添加偏置。
<<<<<<< HEAD
        # print("卷积init44444444444444444444444444444444444完成")

    def forward(self, x):
        # 连接各个cnn各个模块
        # print("x的shape333333333333333", x.shape)
        e1 = self.conv1(x)
        # print("开始执行第一层卷积e1111111111111111111")
        # print("e1shape", e1.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        # print("xshape:", x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print("x111111111:",x.shape)
        x = x.view(-1, 16*84*84)
        # print("x2222",x.shape)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # 返回运算后的结果
        return x
=======

    def forward(self, x):
        # 连接各个cnn各个模块
        e1 = self.conv1(x)
        print("e1shape", e1.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        print("xshape:", x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        print("x111111111:",x.shape)
        x = x.view(-1, 16*84*84)
        print("x2222",x.shape)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # 返回运算后的结果
        return x


print("运行了将图像进行特征提取向量1111111111111111111111111111运行了image_to_conv.py文件")
>>>>>>> master
