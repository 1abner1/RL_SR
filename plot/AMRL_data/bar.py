import matplotlib.pyplot as plt


labels = ['MAML', 'PPO', 'PEAL', 'AMRL_PEP', 'AMRL_ICM', 'AMRL_Ours']
trget_search_dis1 = [15, 24, 16, 27, 20,4]
trget_search_dis2 = [25, 27, 20, 29, 25,5]
trget_search_dis3 = [28, 30, 25, 32, 27,8]
# map4 = [15, 25, 15, 30, 20,8]
trget_search_dis1_std = [2, 2, 1, 1, 2,1]
trget_search_dis2_std = [2, 2, 2, 2, 2,1]
trget_search_dis3_std = [2, 1, 2, 2, 2,1]

width = 0.45       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

cum =list(map(sum, zip(list(trget_search_dis1),list(trget_search_dis2))))
ax.bar(labels, trget_search_dis1, width, yerr=trget_search_dis1_std, label='dis=30%')
ax.bar(labels, trget_search_dis2, width, yerr=trget_search_dis2_std,label='dis=60%', bottom=trget_search_dis1)
ax.bar(labels, trget_search_dis3, width, yerr=trget_search_dis3_std, label='dis=90%',bottom=cum)

ax.set_ylabel('Number of collisions')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()
plt.savefig("Target search Avoid rate.pdf")
plt.show()