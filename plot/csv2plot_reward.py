import matplotlib.pyplot as plt
import csv
import numpy as np
 
def _smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point     # Calculate smoothed value
        smoothed.append(smoothed_val)                           # Save it
        last = smoothed_val                                     # Anchor the last smoothed value
    return np.array(smoothed)
def reader_csv(csv_path):
    csvfile1 = open(csv_path,'r')
    plots1 = csv.reader(csvfile1, delimiter=',')
    x=[]
    y=[]
    for row in plots1:
        x.append((row[0]))
        y.append((row[1]))
    y = [float(i) for i in y]
    x = [float(i) for i in x]
    return x,y
path1 = r"D:\car_experence\plot\mean_reward\run-a2c_run_with_unity_Car_team=0_log-tag-AGENT_total_rt_mean.csv"
path2 = r"D:\car_experence\plot\mean_reward\run-ddpg_run_with_unity_Car_team=0_log-tag-AGENT_total_rt_mean.csv"
path3 = r"D:\car_experence\plot\mean_reward\run-ppo_run_with_unity_Car_team=0_log-tag-AGENT_total_rt_mean.csv"
path4 = r"D:\car_experence\plot\mean_reward\run-sac_run_with_unity_Car_team=0_log-tag-AGENT_total_rt_mean.csv"
path5 = r"D:\car_experence\plot\mean_reward\run-trpo_run_with_unity_Car_team=0_log-tag-AGENT_total_rt_mean.csv"
path6 = r"D:\car_experence\plot\mean_reward\run-dqn_run_with_unity_Car_team=0_log-tag-AGENT_total_rt_mean.csv"
x1, y1 = reader_csv(path1)
x2, y2 = reader_csv(path2)
x3, y3 = reader_csv(path3)
x4, y4 = reader_csv(path4)
x5, y5 = reader_csv(path5)
x6, y6 = reader_csv(path6)
print(f"x1长度:{len(x1)}|y1长度:{len(y1)}")
print(f"x2长度:{len(x2)}|y2长度:{len(y2)}")
print(f"x3长度:{len(x3)}|y3长度:{len(y3)}")
print(f"x4长度:{len(x4)}|y2长度:{len(y4)}")
print(f"x5长度:{len(x5)}|y2长度:{len(y5)}")
print(f"x5长度:{len(x6)}|y2长度:{len(y6)}")
y1 = y1[0:962]
y2 = y2[0:962]
y3 = y3[0:962]
y4 = y4[0:962]
y5 = y5[0:962]
y6 = y6[0:962]
# print(f"y3的长度为{len(y5)}")
y1 = _smooth(y1,0.993)
y2 = _smooth(y2,0.993)
y3 = _smooth(y3,0.993)
y4 = _smooth(y4,0.993)
y5 = _smooth(y5,0.993)
y6 = _smooth(y6,0.993)
# plt.plot(x3,y1,y2,y3,y4,y5)
plt.plot(x3,y1)
plt.plot(x3,y2)
plt.plot(x3,y3)
plt.plot(x3,y4)
plt.plot(x3,y5)
plt.plot(x3,y6)
plt.xlabel('Steps')
plt.ylabel('mean_reward')
plt.legend(loc='upper right',labels=['a2c','ddpg','ppo','sac','trpo'])
plt.show()
