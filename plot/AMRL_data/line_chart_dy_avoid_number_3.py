import matplotlib.pyplot as plt

#static avoid number 1,dy_number 1;static avoid number 1,dy_number 1;static avoid number 3,dy_number 2.
x = [100,200,300,400,500,600]#点的横坐标
k1 = [0.210,0.300,0.320,0.350,0.380,0.400]#线1的纵坐标
k2 = [0.256,0.321,0.352,0.410,0.462,0.478]#线2的纵坐标
k3 = [0.245,0.300,0.355,0.466,0.523,0.501]
k4 = [0.215,0.334,0.372,0.452,0.495,0.515]
k5 = [0.241,0.503,0.415,0.508,0.543,0.560]
k6 = [0.322,0.504,0.533,0.5290,0.615,0.675]
k7 = [0.388,0.511,0.522,0.553,0.625,0.645]
k8 = [0.391,0.579,0.612,0.620,0.635,0.655]
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
plt.savefig("static obs line chart static1 dy3.pdf")
plt.show()
