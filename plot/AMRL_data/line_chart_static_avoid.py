import matplotlib.pyplot as plt

#Target search sucess rate
#折线图
x = [100,200,300,400,500,600]#点的横坐标
k1 = [0.272,0.308,0.504,0.556,0.677,0.775]#线1的纵坐标
k2 = [0.318,0.503,0.603,0.660,0.695,0.785]#线2的纵坐标
k3 = [0.428,0.613,0.743,0.750,0.805,0.815]
k4 = [0.398,0.553,0.723,0.725,0.755,0.825]
k5 = [0.358,0.523,0.653,0.710,0.735,0.805]
k6 = [0.408,0.604,0.785,0.790,0.835,0.825]
k7 = [0.468,0.643,0.753,0.810,0.845,0.845]
k8 = [0.488,0.674,0.765,0.800,0.865,0.955]
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
plt.savefig("static obs line chart sucess rate1.pdf")
plt.show()
