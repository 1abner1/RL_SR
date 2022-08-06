import matplotlib.pyplot as plt

#Target search sucess rate
#折线图
x = [100,200,300,400,500,600]#点的横坐标
k1 = [0.272,0.288,0.404,0.456,0.537,0.635]#线1的纵坐标
k2 = [0.358,0.403,0.543,0.640,0.705,0.845]#线2的纵坐标
k3 = [0.308,0.343,0.603,0.740,0.755,0.865]
k4 = [0.408,0.503,0.703,0.755,0.805,0.925]
k5 = [0.388,0.553,0.663,0.760,0.855,0.965]
k6 = [0.368,0.534,0.635,0.780,0.895,0.955]
k7 = [0.408,0.653,0.863,0.850,0.915,0.945]
k8 = [0.528,0.734,0.835,0.940,0.955,0.995]
plt.plot(x,k1,'s-',color = 'r',label="MAML")#s-:方形
plt.plot(x,k2,'o:',color = 'g',label="PEAL")#o-:圆形
plt.plot(x,k3,'^-.',color = 'b',label="RL^2")#s-:方形
plt.plot(x,k4,'p-',color = 'c',label="EPG")#o-:圆形
plt.plot(x,k5,'+-',color = 'm',label="PPO")#s-:方形
plt.plot(x,k6,'*-',color = 'k',label="AMRL-PEP")#o-:圆形
plt.plot(x,k7,'h--',color = 'y',label="AMRL-ICM")#s-:方形
plt.plot(x,k8,'d-',color = '#008000',label="AMRL")#o-:圆形
plt.xlabel("Episode length")#横坐标名字
plt.ylabel("Success rate")#纵坐标名字
plt.legend(loc = "best")#图例
plt.savefig("Target search sucess rate1.pdf")
plt.show()
