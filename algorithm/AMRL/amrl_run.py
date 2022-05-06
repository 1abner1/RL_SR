# run amrl environment
# install python=3.9
#pip install mlagents==0.25.0
#pip install torch gym numpy==1.20.3
#4.使用cuda 10.2  pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
import numpy as np
import metalearn as ml
import logging
import itertools
from Unity_Env_API.unity_wrapper import UnityWrapper
from Image_deal.image_to_conv import CNNNet
import torch
import argparse
import algorithm.s2rlog.makelog as mlog
from algorithm.AMRL.sorft_attention.soft_attention import soft_attention_net

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    args = parser.parse_known_args()[0]
    return args

def ada_meta_rl():
    meta_learn = ml.MetaLearner()

def meta_train():
   # 创建30种环境，创建一个字典一个数字对应一个任务
   # env = unity_wrapper.UnityWrapper(train_mode=True, base_port=5004,file_name=r"D:\Pytorch_RL_SR\algorithm\AMRL\testtask\car_seg_avoid.exe")
   task_pool={"task1":"D:/Pytorch_RL_SR/algorithm\AMRL/task1/car_seg_avoid.exe",
              "task2":"D:/Pytorch_RL_SR/algorithm\AMRL/task2/car_seg_avoid.exe",
              "task3":"D:/Pytorch_RL_SR/algorithm\AMRL/task3/car_seg_avoid.exe",
              "task4":"D:/Pytorch_RL_SR/algorithm\AMRL/task4/car_seg_avoid.exe",
              "task5":"D:/Pytorch_RL_SR/algorithm\AMRL/task5/car_seg_avoid.exe",
              "task6":"D:/Pytorch_RL_SR/algorithm\AMRL/task6/car_seg_avoid.exe",
              "task7":"D:/Pytorch_RL_SR/algorithm\AMRL/task7/car_seg_avoid.exe",
              "task8":"D:/Pytorch_RL_SR/algorithm\AMRL/task8/car_seg_avoid.exe",
              "task9":"D:/Pytorch_RL_SR/algorithm\AMRL/task9/car_seg_avoid.exe",
              "task10":"D:/Pytorch_RL_SR/algorithm\AMRL/task10/car_seg_avoid.exe",
              "task11":"D:/Pytorch_RL_SR/algorithm\AMRL/task11/car_seg_avoid.exe",
              "task12":"D:/Pytorch_RL_SR/algorithm\AMRL/task12/car_seg_avoid.exe",
              "task13":"D:/Pytorch_RL_SR/algorithm\AMRL/task13/car_seg_avoid.exe",
              "task14":"D:/Pytorch_RL_SR/algorithm\AMRL/task14/car_seg_avoid.exe",
              "task15":"D:/Pytorch_RL_SR/algorithm\AMRL/task15/car_seg_avoid.exe",
              "task16":"D:/Pytorch_RL_SR/algorithm\AMRL/task16/car_seg_avoid.exe",
              "task17":"D:/Pytorch_RL_SR/algorithm\AMRL/task17/car_seg_avoid.exe",
              "task18":"D:/Pytorch_RL_SR/algorithm\AMRL/task18/car_seg_avoid.exe",
              "task19":"D:/Pytorch_RL_SR/algorithm\AMRL/task19/car_seg_avoid.exe",
              "task20":"D:/Pytorch_RL_SR/algorithm\AMRL/task20/car_seg_avoid.exe",
              "task21":"D:/Pytorch_RL_SR/algorithm\AMRL/task21/car_seg_avoid.exe",
              "task22":"D:/Pytorch_RL_SR/algorithm\AMRL/task22/car_seg_avoid.exe",
              "task23":"D:/Pytorch_RL_SR/algorithm\AMRL/task23/car_seg_avoid.exe",
              "task24":"D:/Pytorch_RL_SR/algorithm\AMRL/task24/car_seg_avoid.exe",
              "task25":"D:/Pytorch_RL_SR/algorithm\AMRL/task25/car_seg_avoid.exe",
              "task26":"D:/Pytorch_RL_SR/algorithm\AMRL/task26/car_seg_avoid.exe",
              "task27":"D:/Pytorch_RL_SR/algorithm\AMRL/task27/car_seg_avoid.exe",
              "task28":"D:/Pytorch_RL_SR/algorithm\AMRL/task28/car_seg_avoid.exe",
              "task29":"D:/Pytorch_RL_SR/algorithm\AMRL/task29/car_seg_avoid.exe",
              "task30":"D:/Pytorch_RL_SR/algorithm\AMRL/task30/car_seg_avoid.exe"
              }
   print(task_pool["task1"])


def perceaction(self,state):
    # 同质感知数据处理（将三个图像摄像头数据通过卷积层）
    forward_image = CNNNet()
    net = CNNNet()
    OUTPUT_obs = net.forward(state)
    out_obs_array = OUTPUT_obs[0]
    out_obs_array = out_obs_array.detach().numpy()
    input_obs_state = torch.from_numpy(out_obs_array)

    return input_obs_state


def sorft_attention():
    soft_attention_net1=soft_attention_net()


def main():
    # 制作虚实结合的log 文件
    mlog1 = mlog.run()
    # logging.basicConfig(level=logging.INFO)
    # 获得参数信息
    parmater = get_args()
    env = UnityWrapper(train_mode=True, base_port=5004,file_name=r"D:\Pytorch_RL_SR\algorithm\AMRL\testtask\car_seg_avoid.exe")
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    state_dim = obs_shape_list
    print("状态维度：",state_dim)

    #train makeure task
    for episode in range(2):
        obs_list = env.reset
        # print("开始运行到下一步")
        print("获得观察值",obs_list)
        for step in range(2):
            print("环境没有问题")


if __name__ == "__main__":
    main()
