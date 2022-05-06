import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
x = np.random.choice(255,(84,84,3))
print("x",x)
x = np.uint8(x)
# plt.imshow(x)
# plt.show(block=False)
# plt.pause(2)
# plt.close()
PIL_image = Image.fromarray(x)
plt.imshow(PIL_image)
plt.show(block=False)
plt.pause(2)
plt.close()
transformer = transforms.Compose([transforms.Resize((84, 84)),  # resize 调整图片大小
                                        # transforms.RandomHorizontalFlip(), # 水平反转
                                        transforms.ToTensor(),  # 0-255 to 0-1 归一化处理
                                        transforms.Normalize(mean=(0,0,0),std=(1,1,1)) #归一化
                                        ])
x=transformer(PIL_image)
print("归一化之后的图像",x)