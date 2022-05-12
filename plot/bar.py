import matplotlib.pyplot as plt


labels = ['PPO', 'SAC', 'TRPO', 'A3C', 'DDPG', 'AMDDPG']
map1 = [15, 24, 16, 27, 20,4]
map2 = [25, 27, 20, 29, 25,5]
map3 = [28, 30, 25, 32, 27,8]
# map4 = [15, 25, 15, 30, 20,8]
map1_std = [2, 2, 1, 1, 2,1]
map2_std = [2, 2, 2, 2, 2,1]   
map3_std = [2, 1, 2, 2, 2,1]

width = 0.45       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

cum =list(map(sum, zip(list(map1),list(map2))))
ax.bar(labels, map1, width, yerr=map1_std, label='map1')
ax.bar(labels, map2, width, yerr=map2_std,label='map2', bottom=map1)
ax.bar(labels, map3, width, yerr=map3_std, label='map3',bottom=cum)

ax.set_ylabel('Number of collisions')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()

plt.show()