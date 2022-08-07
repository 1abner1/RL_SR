import matplotlib.pyplot as plt


labels = ['MAML', 'PPO', 'PEAL', 'AMRL_PEP', 'AMRL_ICM', 'AMRL_Ours']
static_aboid_number1 = [15, 24, 16, 25, 20,4]
static_aboid_number2 = [24, 23, 20, 29, 24,4]
static_aboid_number3 = [30, 27, 25, 30, 23,6]
# map4 = [15, 25, 15, 30, 20,8]
static_aboid_number1_std = [2, 2, 1, 1, 2,1]
static_aboid_number2_std = [2, 2, 2, 2, 2,1]
static_aboid_number3_std = [2, 1, 2, 2, 2,1]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

cum =list(map(sum, zip(list(static_aboid_number1),list(static_aboid_number2))))
ax.bar(labels, static_aboid_number1, width, color ='purple', yerr=static_aboid_number1_std, label='static obs num=1')
ax.bar(labels, static_aboid_number2, width, color ='gray', yerr=static_aboid_number2_std,label='static obs num=2', bottom=static_aboid_number1)
ax.bar(labels, static_aboid_number3, width, color ='orange', yerr=static_aboid_number3_std, label='static obs num=3',bottom=cum)

ax.set_ylabel('Number of collisions')
ax.set_xlabel("Algorithm")
# ax.set_title('Scores by group and gender')
ax.legend()
plt.savefig("static obs Avoid rate.pdf")
plt.show()