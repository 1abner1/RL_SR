import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成圆的参数
theta = np.linspace(0, 2 * np.pi, 100)
r = 0.5

# 生成五个圆的中心点
centers = np.random.uniform(low=-1, high=1, size=(5, 3))

# 创建 3D 坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# 生成并绘制五个圆孔
for i in range(5):
    center = centers[i]
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = center[2] * np.ones_like(x)
    ax.plot(x, y, z, 'b')
    ax.plot_surface(center[0] + r * np.sin(theta)[:, np.newaxis], center[1] + r * np.cos(theta)[:, np.newaxis],
                     center[2] * np.ones_like(theta)[:, np.newaxis], color='w', alpha=0.8)

plt.show()
