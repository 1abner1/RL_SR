import matplotlib.pyplot as plt

#static avoid number 1,dy_number 1;static avoid number 1,dy_number 1;static avoid number 3,dy_number 2.
x = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]#点的横坐标
k3 = [0.707,0.670,0.650,0.560,0.520,0.500,0.470,0.4500,0.400,0.350]#线1的纵坐标
k2 = [0.800,0.750,0.760,0.740,0.770,0.690,0.670,0.650,0.590,0.550]#线2的纵坐标
k1 = [0.90,0.930,0.920,0.940,0.950,0.900,0.890,0.910,0.945,0.978]

plt.plot(x,k1,'s-',color = 'r',label="Random")#s-:方形
plt.plot(x,k2,'+-',color = 'm',label="SAC")#s-:方形
plt.plot(x,k3,'d-',color = '#008000',label="DDCL")#o-:圆形
plt.xlabel("Episode")#横坐标名字
plt.ylabel("Error Rate")#纵坐标名字
plt.legend(loc = "best")#图例
plt.savefig("Error Rate.pdf")
plt.show()
