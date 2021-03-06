# run amrl environment
# install python=3.9
#pip install mlagents==0.29.0
#pip install torch gym numpy==1.20.3
#4.使用cuda 10.2  pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# https://github.com/Unity-Technologies/ml-agents
#pip install opencv-python
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
from image_show.image_show import Unity_image_show
from algorithm.AMRL.Image_deal.image_to_conv import image
import torch.nn as nn
from GruActorCritic.gru import GRUActorCritic
from ppocar.PPO import PPO
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  #记录log
from ray_show.Ray_show import Ray_show

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    args = parser.parse_known_args()[0]
    return args

def Agent_Car():
    num_actions = 2
    num_feature = 8
    ppo_epochs = 32
    mini_batchsize = 8
    model_Gru = GRUActorCritic(num_actions, num_feature)
    optimizer = optim.Adam(model_Gru.parameters(), lr=0.003)
    Car_agent = PPO(model_Gru, optimizer, ppo_epochs, mini_batchsize, batchsize, clip_param, vf_coef, ent_coef, max_grad_norm, target_kl)


def ada_meta_rl():
    # instance_meta_learn =meta_learn()
    # device, model, num_workers, task, num_actions, num_states, num_tasks, num_traj, traj_len, gamma, tau
    # 元学习器，主要引入了
    num_workers = 1
    task = "Car_Object_Search" # "Car_Object_Search",Car_Static_Search , Car_Dynamic_avoid , 三种任务
    num_actions = 2  #(direcation,accleration)
    num_states = 8  #(同质感知数据，异质感知数据，位置信息统一处理成8维)
    num_tasks = 30  # (总任务数)
    num_traj = 1
    traj_len = 1
    gamma = 0.99
    tau = 0.95   #'GAE parameter
    num_feature = num_actions + num_states
    model_Gru = GRUActorCritic(num_actions, num_feature)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_learn = ml.MetaLearner(device,model_Gru,num_workers,num_actions,num_states,num_tasks,num_traj,traj_len,gamma,tau)

def outloop_select_task():
    task_pool = {"task1": "D:/Pytorch_RL_SR/algorithm\AMRL/task1/car_seg_avoid.exe",
                 "task2": "D:/Pytorch_RL_SR/algorithm\AMRL/task2/car_seg_avoid.exe",
                 "task3": "D:/Pytorch_RL_SR/algorithm\AMRL/task3/car_seg_avoid.exe",
                 "task4": "D:/Pytorch_RL_SR/algorithm\AMRL/task4/car_seg_avoid.exe",
                 "task5": "D:/Pytorch_RL_SR/algorithm\AMRL/task5/car_seg_avoid.exe",
                 "task6": "D:/Pytorch_RL_SR/algorithm\AMRL/task6/car_seg_avoid.exe",
                 "task7": "D:/Pytorch_RL_SR/algorithm\AMRL/task7/car_seg_avoid.exe",
                 "task8": "D:/Pytorch_RL_SR/algorithm\AMRL/task8/car_seg_avoid.exe",
                 "task9": "D:/Pytorch_RL_SR/algorithm\AMRL/task9/car_seg_avoid.exe",
                 "task10": "D:/Pytorch_RL_SR/algorithm\AMRL/task10/car_seg_avoid.exe",
                 "task11": "D:/Pytorch_RL_SR/algorithm\AMRL/task11/car_seg_avoid.exe",
                 "task12": "D:/Pytorch_RL_SR/algorithm\AMRL/task12/car_seg_avoid.exe",
                 "task13": "D:/Pytorch_RL_SR/algorithm\AMRL/task13/car_seg_avoid.exe",
                 "task14": "D:/Pytorch_RL_SR/algorithm\AMRL/task14/car_seg_avoid.exe",
                 "task15": "D:/Pytorch_RL_SR/algorithm\AMRL/task15/car_seg_avoid.exe",
                 "task16": "D:/Pytorch_RL_SR/algorithm\AMRL/task16/car_seg_avoid.exe",
                 "task17": "D:/Pytorch_RL_SR/algorithm\AMRL/task17/car_seg_avoid.exe",
                 "task18": "D:/Pytorch_RL_SR/algorithm\AMRL/task18/car_seg_avoid.exe",
                 "task19": "D:/Pytorch_RL_SR/algorithm\AMRL/task19/car_seg_avoid.exe",
                 "task20": "D:/Pytorch_RL_SR/algorithm\AMRL/task20/car_seg_avoid.exe",
                 "task21": "D:/Pytorch_RL_SR/algorithm\AMRL/task21/car_seg_avoid.exe",
                 "task22": "D:/Pytorch_RL_SR/algorithm\AMRL/task22/car_seg_avoid.exe",
                 "task23": "D:/Pytorch_RL_SR/algorithm\AMRL/task23/car_seg_avoid.exe",
                 "task24": "D:/Pytorch_RL_SR/algorithm\AMRL/task24/car_seg_avoid.exe",
                 "task25": "D:/Pytorch_RL_SR/algorithm\AMRL/task25/car_seg_avoid.exe",
                 "task26": "D:/Pytorch_RL_SR/algorithm\AMRL/task26/car_seg_avoid.exe",
                 "task27": "D:/Pytorch_RL_SR/algorithm\AMRL/task27/car_seg_avoid.exe",
                 "task28": "D:/Pytorch_RL_SR/algorithm\AMRL/task28/car_seg_avoid.exe",
                 "task29": "D:/Pytorch_RL_SR/algorithm\AMRL/task29/car_seg_avoid.exe",
                 "task30": "D:/Pytorch_RL_SR/algorithm\AMRL/task30/car_seg_avoid.exe"
                 }
    select_task = task_pool["task1"]

    return select_task

def meta_train():
   # 创建30种环境，创建一个字典一个数字对应一个任务
   # env = unity_wrapper.UnityWrapper(train_mode=True, base_port=5004,file_name=r"D:\Pytorch_RL_SR\algorithm\AMRL\testtask\car_seg_avoid.exe")
   pass

def perceaction_image(state_image):
    state_obs_image = image(state_image)
    return state_obs_image

def sorft_attention():
    soft_attention_net1=soft_attention_net()

def ray_deal(state_ray):
    # 雷达数据处理
    lay = nn.Linear(404, 8)
    # state_ray = state_ray.detach()
    state_ray_tensor = torch.from_numpy(state_ray)
    state_ray1 = lay(state_ray_tensor)
    state_ray_output = state_ray1.detach()
    return state_ray_output

def position_deal(state_posi):
    state_posi = torch.from_numpy(state_posi)
    state_posi1 = state_posi
    return state_posi1

def fusion_sensor_date(obs_list):
    # ---------------视觉信息----------------------
    forward_image = obs_list[0][0]
    left_image = obs_list[1][0]
    right_image = obs_list[2][0]
    # -----------------雷达射线-----------------------
    ray = obs_list[3][0]
    # print("ray 数据",ray)
    # -----------------显示雷达射线--------------------
    # ray_figure = Ray_show(len(ray))
    # ray_figure.show(ray)
    # -----------------向量信息-----------------------
    position = obs_list[4][0]
    # -----------------显示图片-----------------------
    Unity_image_show("forward-left-right",forward_image,left_image,right_image)
    # ------------把图像数据提取特征变成一个8维的向量------
    forward_image_deal_8v = perceaction_image(forward_image)
    left_image_deal_8v = perceaction_image(left_image)
    right_image_deal_8v = perceaction_image(right_image)
    print("forward处理的数据", forward_image_deal_8v)
    print("left_image_deal_8v处理的数据", left_image_deal_8v)
    print("right_image_deal_8v处理的数据", right_image_deal_8v)
    # ---------------雷达射线数据处理成8维--------------
    ray_output = ray_deal(ray)
    print("输出雷达数据", ray_output)
    # -----------------位置向量数据--------------------
    position_output = position_deal(position)
    print("位置向量输出", position_output)
    # ---------------------------感知层---------------------------------
    # ------------------------融合感知数据------------------
    w1 = torch.tensor(0.6, dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    w3 = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    a1 = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)
    a2 = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
    a3 = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
    fusion_same_image = w1 * forward_image_deal_8v + w2 * left_image_deal_8v + w3 * right_image_deal_8v
    fusion_dif_sensor = a1 * forward_image_deal_8v + a2 * ray_output + a3 * position_output + a3 * position_output
    total_fusion = fusion_same_image + fusion_dif_sensor
    # print("fusion_same_image", fusion_same_image)
    # print("fusion_dif_sensor", fusion_dif_sensor)
    # print("total_fusion", total_fusion)

    return total_fusion
    # ------------------------融合感知数据------------------

def random_move():
    # n_agents = obs_list[0].shape[0]
    # for j in range(100):
    #     d_action, c_action = None, None
    #     n_agents = 1
    #     if d_action_size:
    #         d_action = np.random.randint(0, d_action_size, size=n_agents)
    #         d_action = np.eye(d_action_size, dtype=np.int32)[d_action]
    #     if c_action_size:
    #         c_action = np.random.randn(n_agents, c_action_size)
    #     obs_list, reward, done, max_step = env.step(d_action, c_action)  # 环境step
    pass


def main():
    # -----------制作虚实结合的log文件--------------
    show_SR_figure = mlog.run()
    # -----------记录奖励函数曲线-------------------
    reword_log = SummaryWriter('./car')
    # -----------打印unity相关的信息---------------
    logging.basicConfig(level=logging.INFO)
    # -----------获得参数信息----------------------
    parmater = get_args()  #这个还不太会使用
    # par1 = parmater("--task")
    env = UnityWrapper(train_mode=True, base_port=5004,file_name=r"D:\RL_SR\envs\test\car_seg_avoid.exe")
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    state_dim = obs_shape_list
    print("总的状态维度：",state_dim) #(前摄像头图像，左摄像头图像，右摄像头图像，射线数据，目标位置和速度)
    print("前摄像头维度：", state_dim[0])
    print("左摄像头维度：", state_dim[1])
    print("右摄像头维度：", state_dim[2])
    print("雷达维度：", state_dim[3])
    print("向量维度：", state_dim[4])

    #train makeure task
    for episode in range(100):
        #---------------------------感知层--------------------------------
        obs_list = env.reset()
        total_fusion_sensor_date = fusion_sensor_date(obs_list)
        print("total_fusion_sensor_date",total_fusion_sensor_date)
        c_action = np.random.randn(1, 2)
        print("c_action",c_action)
        d_action = None
        obs_list, reward, done, max_step = env.step(d_action, c_action)
        for step in range(2):
            # print("环境没有问题")
            pass

if __name__ == "__main__":
    main()
