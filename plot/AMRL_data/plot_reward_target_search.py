#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid") #这是seaborn默认的风格
#数据处理方法
#1、数据的smooth方法
def smooth(data, sm=5):
    '''
    :param data:
    :param sm: sm表示滑动窗口大小,为2*k+1,
    :return: smooth data
    '''
    smooth_data = []
    if sm > 1:
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
    return smooth_data

def smoothing(data, sm=5):
    '''
    :param data:
    :param sm: sm表示滑动窗口大小,
    :return: smooth data
    '''
    data_smooth = []
    for i,j in enumerate(data):
        index_left = i - int(sm / 2) if (i - int(sm / 2)) >= 0 else 0
        index_right = i + int(sm / 2) if (i + int(sm / 2)) < len(data) else len(data) -1
        data_smooth.append(sum(data[index_left:index_right])/len(data[index_left:index_right]))
    return data_smooth

#2、数据max min mean
def get_max_min_mean(data, sm):
    '''
    :param data:
    :param sm:
    :return:
    '''
    data_max = []
    data_min = []
    data_mean = []
    for i,j in enumerate(data):
        index_left = i - int(sm / 2) if (i - int(sm / 2)) >= 0 else 0
        index_right = i + int(sm / 2) if (i + int(sm / 2)) < len(data) else len(data) -1
        data_max.append(max(data[index_left:index_right]))
        data_min.append(min(data[index_left:index_right]))
        data_mean.append(sum(data[index_left:index_right])/len(data[index_left:index_right]))
    return data_max, data_min, data_mean

#画一条线的line和scale
def draw_line(data_file, label, color, max_min_mean_sm=6, smooth_sm=40):
    #数据处理
    #Wall time	Step	Value
    data = pd.read_csv(filepath_or_buffer=data_file)
    #DDPG_predator_4.head(10)
    # data_x = data['Step']
    data_x = data['Step'][0:600]
    # data_y = data['Value']
    data_y = data['Value'][0:600]
    data_y = (data_y+1.5)/6.5

    data_max_y, data_min_y, data_mean_y \
        = get_max_min_mean(data_y, sm=max_min_mean_sm)

    data_max_smooth_y = smoothing(data=data_max_y, sm=smooth_sm)
    data_min_smooth_y = smoothing(data=data_min_y, sm=smooth_sm)
    data_mean_smooth_y = smoothing(data=data_mean_y, sm=smooth_sm)
    #print(data_mean_y)
    #画图
    plt.plot(data_x, data_mean_smooth_y, color = color, label=label, linewidth='2')

    plt.fill_between(data_x,
                     data_min_smooth_y,
                     data_mean_smooth_y,
                     facecolor=color,
                     alpha=0.1)
    plt.fill_between(data_x,
                     data_mean_smooth_y,
                     data_max_smooth_y,
                     facecolor=color,
                     alpha=0.1)



def curve_plot():
    plt.figure()

    data_files_path = [r'D:\RL_SR\plot\AMRL_data\Target search reward\run-sac_car_unknow_target_place3_carconavoid-tag-Environment_Cumulative Reward (1).csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-sac_car_unknow_target_place_carconavoid-tag-Environment_Cumulative Reward.csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-poca_car_unknow_target_place1_carconavoid-tag-Environment_Cumulative Reward.csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-poca_car_unknow_target_place2_carconavoid-tag-Environment_Cumulative Reward.csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-poca_car_unknow_target_place3_carconavoid-tag-Environment_Cumulative Reward.csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-poca_car_unknow_target_place_carconavoid-tag-Environment_Cumulative Reward.csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-car_unknow_target_place3_carconavoid-tag-Environment_Cumulative Reward.csv',
                       r'D:\RL_SR\plot\AMRL_data\Target search reward\run-carconavoid5_carconavoid-tag-Environment_Cumulative Reward.csv'
                       ]

    labels = ['MAML',
              'EPG',
              'RL2',
              'PEAL',
              'PPO',
              'AMRL_PEP',
              'AMRL_ICM',
              'AMR(OURS)'
              ]

    colors = ['r','g','m','y','olive','c','orange','b']

    smooth_sms = [2,2,2,2,2,2,2,2]

    for data_file_path, label, color, smooth_sm in zip(data_files_path, labels, colors, smooth_sms):
        draw_line(data_file=data_file_path,
                  label=label,
                  color=color,
                  max_min_mean_sm=20,
                  smooth_sm=smooth_sm)

    #figure的具体设置需要在直线等画完了在进行
    plt.xlabel("Step")#横坐标名字
    plt.ylabel("Reward")#纵坐标名字
    #plt.legend(loc="best")#图例
    plt.legend(loc="lower right")#图例
    plt.ylim(0,1)
    plt.xlim(0, 300000)
    plt.savefig("target search reward1.pdf")
    plt.show()


curve_plot()