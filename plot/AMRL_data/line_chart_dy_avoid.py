import matplotlib.pyplot as plt


#折线图
x = [100,200,300,400,500,600]#点的横坐标
k1 = [0.272,0.348,0.504,0.526,0.537,0.545]#线1的纵坐标
k2 = [0.318,0.653,0.743,0.745,0.765,0.735]
k3 = [0.308,0.593,0.723,0.730,0.755,0.725]
k4 = [0.318,0.653,0.743,0.745,0.765,0.735]
k5 = [0.298,0.623,0.653,0.725,0.745,0.715]
k6 = [0.398,0.734,0.779,0.780,0.805,0.765]
k7 = [0.408,0.743,0.793,0.800,0.815,0.815]
k8 = [0.438,0.754,0.805,0.830,0.845,0.855]
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
plt.savefig("dy obs line chart sucess rate1.pdf")
plt.show()
