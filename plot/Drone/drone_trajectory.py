import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建三维图形对象
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 定义四条轨迹的坐标点
traj1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])
traj2 = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4], [5, 0, 5], [6, 0, 6], [7, 0, 7], [8, 0, 8], [9, 0, 9]])
traj3 = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4], [0, 5, 5], [0, 6, 6], [0, 7, 7], [0, 8, 8], [0, 9, 9]])
traj4 = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0], [5, 5, 0], [6, 6, 0], [7, 7, 0], [8, 8, 0], [9, 9, 0]])

# 绘制四条轨迹
# 第一条轨迹为红色实线
xs = traj1[:,0]
ys = traj1[:,1]
zs = traj1[:,2]
ax.plot(xs, ys, zs, color='r', linestyle='solid', label='Trajectory 1')

# 第二条轨迹为绿色虚线
xs = traj2[:,0]
ys = traj2[:,1]
zs = traj2[:,2]
ax.plot(xs, ys, zs, color='g', linestyle='dashed', label='Trajectory 2')

# 第三条轨迹为蓝色点线
xs = traj3[:,0]
ys = traj3[:,1]
zs = traj3[:,2]
ax.plot(xs, ys, zs, color='b', linestyle='dotted', label='Trajectory 3')

# 第四条轨迹为黑色实线
xs = traj4[:,0]
ys = traj4[:,1]
zs = traj4[:,2]
ax.plot(xs, ys, zs, color='k', linestyle='solid', label='Trajectory 4')

# 设置图例和图形和子图标题和标签
ax.legend()
ax.set_title('Indoor UAV Trajectory')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.tight_layout(rect=[0, 0, 1, 0.5])
# print
plt.savefig('indoor_uav_trajectory2.pdf')
plt.show()


