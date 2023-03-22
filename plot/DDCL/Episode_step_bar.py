import matplotlib.pyplot as plt


labels = ['Random', 'SAC', 'DDCL']
map1 = [1000, 500, 100]
# colors = ['r','g','m','y','olive','c','orange','b']
width = 0.50     # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()


ax.bar(labels, map1, width)
# c.set_color(['r', 'r', 'b', 'r','b','b'])
ax.get_children()
ax.get_children()[0].set_color('purple')
ax.get_children()[1].set_color('orange')
ax.get_children()[2].set_color('cyan')


ax.set_ylabel('Steps')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()
plt.savefig("Episode_step_bar.pdf")
plt.show()