#奖励函数曲线
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def getdata():
    basecond = [[18, 20, 19, 18, 13, 4, 1],
                [20, 17, 12, 9, 3, 0, 0],
                [20, 20, 20, 12, 5, 3, 0]]

    cond1 = [[18, 19, 18, 19, 20, 15, 14],
             [19, 20, 18, 16, 20, 15, 9],
             [19, 20, 20, 20, 17, 10, 0],
             [20, 20, 20, 20, 7, 9, 1]]

    cond2 = [[20, 20, 20, 20, 19, 17, 4],
             [20, 20, 20, 20, 20, 19, 7],
             [19, 20, 20, 19, 19, 15, 2]]

    cond3 = [[20, 20, 20, 20, 19, 17, 12],
             [18, 20, 19, 18, 13, 4, 1],
             [20, 19, 18, 17, 13, 2, 0],
             [19, 18, 20, 20, 15, 6, 0]]

    return basecond, cond1, cond2, cond3

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return smooth_data

if __name__ == '__main__':
    data = getdata()
    fig = plt.figure()

    xdata = np.array([0, 1, 2, 3, 4, 5, 6])/5
    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    label = ['algo1', 'algo2', 'algo3', 'algo4']

    for i in range(4):
        sns.lmplot(xdata, data=data[i], color=color[i], linestyle=linestyle[i], condition=label[i])

    df = pd.DataFrame(dict(time=np.arange(500),
                           value=np.random.randn(500).cumsum()))
    # g = sns.relplot(x="time", y="value", kind="line", data=df)
    # sns.lineplot()

    plt.ylabel("Success Rate", fontsize=25)
    plt.xlabel("Iteration Number", fontsize=25)
    plt.title("Awesome Robot Performance", fontsize=30)

    plt.legend(loc='bottom left')
    plt.savefig("/home/zhr/Pictures/img1.png")
    plt.show()