import matplotlib.pyplot as plt


labels = ['MAML', 'PEAL', 'RL2', 'EPG', 'PPO', 'AMRL_P','AMRL_I','AMRL']
map1 = [24, 17, 20, 19, 21, 15, 16, 9]
# colors = ['r','g','m','y','olive','c','orange','b']
width = 0.45       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()


ax.bar(labels, map1, width)
# c.set_color(['r', 'r', 'b', 'r','b','b'])
ax.get_children()
ax.get_children()[0].set_color('r')
ax.get_children()[1].set_color('g')
ax.get_children()[2].set_color('m')
ax.get_children()[3].set_color('y')
ax.get_children()[4].set_color('olive')
ax.get_children()[5].set_color('c')
ax.get_children()[6].set_color('orange')
ax.get_children()[7].set_color('b')

ax.set_ylabel('Train Time(h)')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()
plt.savefig("target_search_train_time.pdf")
plt.show()