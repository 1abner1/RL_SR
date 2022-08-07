import matplotlib.pyplot as plt


labels = ['MAML', 'PEAL', 'RL2','EPG', 'PPO','AMRL_P', 'AMRL_I', 'AMRL']
trget_search_dis1 = [17, 21, 16,17, 19, 21, 22,5]
trget_search_dis2 = [27, 25, 21,19, 18, 25, 21,6]
trget_search_dis3 = [32, 30, 28,27, 20, 35, 32,10]
# map4 = [15, 25, 15, 30, 20,8]
trget_search_dis1_std = [2, 2, 1, 1,1, 1, 2,1]
trget_search_dis2_std = [2, 2, 2, 2, 2,2, 2,1]
trget_search_dis3_std = [2, 1, 2, 2, 2,2, 2,1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

cum =list(map(sum, zip(list(trget_search_dis1),list(trget_search_dis2))))
ax.bar(labels, trget_search_dis1, width, color ='purple', yerr=trget_search_dis1_std, label='static obs num=1')
ax.bar(labels, trget_search_dis2, width, color ='gray', yerr=trget_search_dis2_std,label='static obs num=2', bottom=trget_search_dis1)
ax.bar(labels, trget_search_dis3, width, color ='orange', yerr=trget_search_dis3_std, label='static obs num=3',bottom=cum)

ax.set_ylabel('Number of collisions')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()
plt.savefig("Target search static Avoid rate bar2.pdf")
plt.show()