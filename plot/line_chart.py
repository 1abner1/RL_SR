import matplotlib.pyplot as plt

#折线图
x = [5,7,11,17,19,25]#点的横坐标
k1 = [0.1222,0.218,0.3344,0.4262,0.5371,0.8353]#线1的纵坐标
k2 = [0.1588,0.2334,0.5435,0.6407,0.7453,0.9453]#线2的纵坐标
k3 = [0.2088,0.3034,0.6035,0.8007,0.9053,0.9653]
k4 = [0.4088,0.5034,0.7035,0.7357,0.8053,0.8853]
k5 = [0.3888,0.5534,0.6635,0.7507,0.8053,0.9853]
k6 = [0.3688,0.4934,0.5935,0.6807,0.7353,0.7853]
k7 = [0.4088,0.6534,0.8635,0.9007,0.9153,0.9553]
k8 = [0.4288,0.634,0.735,0.8007,0.8953,0.9053]
plt.plot(x,k1,'s-',color = 'r',label="PPO")#s-:方形
plt.plot(x,k2,'o:',color = 'g',label="MAML")#o-:圆形
plt.plot(x,k3,'^-.',color = 'b',label="RL^2")#s-:方形
plt.plot(x,k4,'p-',color = 'c',label="AMRL")#o-:圆形
plt.plot(x,k5,'+-',color = 'm',label="ATT-RLSTM")#s-:方形
plt.plot(x,k6,'*-',color = 'k',label="CNN-RLSTM")#o-:圆形
plt.plot(x,k7,'h--',color = 'y',label="ATT-RLSTM")#s-:方形
plt.plot(x,k8,'d-',color = '#008000',label="CNN-RLSTM")#o-:圆形
plt.xlabel("Episode length")#横坐标名字
plt.ylabel("Success rate")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
