import matplotlib.pyplot as plt

#static avoid number 1,number 2,number 3.
x = [100,200,300,400,500,600]#点的横坐标
k1 = [0.371,0.492,0.522,0.531,0.549,0.597]#线1的纵坐标
k2 = [0.334,0.461,0.560,0.620,0.595,0.615]#线2的纵坐标
k3 = [0.378,0.523,0.513,0.609,0.675,0.642]
k4 = [0.391,0.564,0.572,0.592,0.615,0.675]
k5 = [0.478,0.593,0.624,0.660,0.705,0.755]
k6 = [0.468,0.585,0.614,0.643,0.695,0.725]
k7 = [0.548,0.653,0.693,0.710,0.705,0.755]
k8 = [0.428,0.704,0.732,0.740,0.755,0.815]
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
plt.savefig("static obs line chart sucess rate1 number3.pdf")
plt.show()
