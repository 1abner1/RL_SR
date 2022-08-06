import matplotlib.pyplot as plt

#static avoid number 1,dy_number 1;static avoid number 1,dy_number 1;static avoid number 3,dy_number 2.
x = [100,200,300,400,500,600]#点的横坐标
k1 = [0.372,0.401,0.442,0.433,0.417,0.450]#线1的纵坐标
k2 = [0.383,0.412,0.428,0.437,0.491,0.509]#线2的纵坐标
k3 = [0.457,0.593,0.582,0.640,0.675,0.685]
k4 = [0.404,0.504,0.522,0.582,0.575,0.595]
k5 = [0.321,0.423,0.505,0.518,0.573,0.650]
k6 = [0.432,0.514,0.563,0.5790,0.625,0.675]
k7 = [0.378,0.531,0.562,0.582,0.615,0.705]
k8 = [0.421,0.554,0.642,0.660,0.685,0.755]
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
plt.savefig("static obs line chart static1 dy2.pdf")
plt.show()
