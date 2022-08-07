import matplotlib.pyplot as plt


labels = ['MAML', 'PEAL', 'RL2','EPG', 'PPO','AMRL_P', 'AMRL_I', 'AMRL']
trget_search_dis1 = [20, 24, 23,22, 25, 26, 25,12]
trget_search_dis2 = [30, 31, 30,35, 34, 32, 30,15]
trget_search_dis3 = [34, 32, 37,35, 36, 40, 45,20]
# map4 = [15, 25, 15, 30, 20,8]
trget_search_dis1_std = [2, 2, 1, 1,1, 1, 2,1]
trget_search_dis2_std = [2, 2, 2, 2, 2,2, 2,1]
trget_search_dis3_std = [2, 1, 2, 2, 2,2, 2,1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

cum =list(map(sum, zip(list(trget_search_dis1),list(trget_search_dis2))))
ax.bar(labels, trget_search_dis1, width, color ='purple', yerr=trget_search_dis1_std, label='sta_obs_num=1 dyn_obs_num=1')
ax.bar(labels, trget_search_dis2, width, color ='gray', yerr=trget_search_dis2_std,label='sta_obs_num=2 dyn_obs_num=1', bottom=trget_search_dis1)
ax.bar(labels, trget_search_dis3, width, color ='orange', yerr=trget_search_dis3_std, label='sta_obs_num=3 dyn_obs_num=2',bottom=cum)

ax.set_ylabel('Number of collisions')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()
plt.savefig("Target search dy Avoid rate bar3.pdf")
plt.show()