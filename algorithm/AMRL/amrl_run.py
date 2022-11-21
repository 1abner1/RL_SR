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
from torch.distributions import MultivariateNormal
import os
import random
import time
from per_expericence.prioritized_memory import Memory
import tqdm

device = torch.device('cpu')


class AMRL_Algorithm():

    def __init__(self,state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        # super(PPO_Algorithm, self).__init__(self,state_dim,action_dim)
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.action_std = action_std_init
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.policy_value = Actor_crtic_network(state_dim,action_dim)
        self.policy_value_old = Actor_crtic_network(state_dim,action_dim)
        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_value.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy_value.critic.parameters(), 'lr': lr_critic}
        ])

    def selection_action(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_value.Actor_policy(state)   #关键点
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.detach().cpu().numpy().flatten()

    def network_update(self):
        # self.states = state
        # self.actions = action
        # self.rewards = reward
        # self.logprobs = action_logsprob
        # self.is_terminals = done

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_value.Critic_value(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # 这个更新方式为clip 的ppo 而不是pential 的ppo (使用kl散度)，目标是让获得评估值和奖励函数进行对比，state_value是通过critic输出的，actor获得动作
            loss2 = abs(state_values - rewards).detach()
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,rewards) - 0.01 * dist_entropy  # dist_entropy 为多元高斯分布的信息熵(传入动作的信息熵) 最后一项表示的增加探索项。
            self.loss = loss
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            loss1 = loss.mean()
        # Copy new weights into old policy
        self.policy_value_old.load_state_dict(self.policy_value.state_dict())
        # clear buffer
        self.buffer.clear()

        return loss1, loss2


    def save_network_parm(self, checkpoint_path):
        torch.save(self.policy_value_old.state_dict(), checkpoint_path)

    def load_network_parm(self, checkpoint_path):
        self.policy_value_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_value.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy_value.set_action_std(new_action_std)
        self.policy_value_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor_crtic_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_crtic_network, self).__init__()
        self.action_dim = action_dim
        self.action_std_init = 0.6
        self.action_var = torch.full((action_dim,), self.action_std_init * self.action_std_init).to(device)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def Actor_policy(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()  # 动作采用多元高斯分布采样
        action_logprob = dist.log_prob(action)  # 这个动作概率就相当于优先经验回放的is_weight
        return action.detach(), action_logprob.detach()

    def Critic_value(self, state, action):
        action_mean = self.actor(state)  # 将状态输入到actor 网络中输出动作action
        action_var = self.action_var.expand_as(action_mean)  # 均值
        cov_mat = torch.diag_embed(action_var).to(device)  # 方差
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)  # 取动作的概率
        dist_entropy = dist.entropy()  # 多元高斯分布的熵是什么，熵是两个分布的比值，熵是否理解为期望，所有值的概率*值，信息熵是一个值
        state_values = self.critic(state)  # 将状态输入到critic 中得到的是状态值
        return action_logprobs, state_values, dist_entropy

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

class Attention(nn.Module):#自定义类 继承nn.Module

    def __init__(self):#初始化函数
        super(Attention, self).__init__()#继承父类初始化函数
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(inplace=True),
        )#自定义实例属性 model 传入自定义模型的内部构造 返回类

    def forward(self, x):
        x = self.model(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.MLP = MLP.MLP()  # 使用soft 注意力机制

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

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
    batchsize = 32
    clip_param =0.01
    vf_coef = 0.001
    ent_coef = 0.002
    max_grad_norm = 0.03
    target_kl = 0.04
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

def outloop_select_task(scene_name):
    task_pool_target = {"task1": r"D:/Pytorch_RL_SR/algorithm/AMRL/task1/car_seg_avoid.exe",
                 "task2": r"D:/Pytorch_RL_SR/algorithm/AMRL/task2/car_seg_avoid.exe",
                 "task3": r"D:/Pytorch_RL_SR/algorithm/AMRL/task3/car_seg_avoid.exe",
                 "task4": r"D:/Pytorch_RL_SR/algorithm/AMRL/task4/car_seg_avoid.exe",
                 "task5": r"D:/Pytorch_RL_SR/algorithm/AMRL/task5/car_seg_avoid.exe",
                  "task6": r"D:/Pytorch_RL_SR/algorithm/AMRL/task6/car_seg_avoid.exe",
                 "task7": r"D:/Pytorch_RL_SR/algorithm/AMRL/task7/car_seg_avoid.exe",
                 "task8": r"D:/Pytorch_RL_SR/algorithm/AMRL/task8/car_seg_avoid.exe",
                 "task9": r"D:/Pytorch_RL_SR/algorithm/AMRL/task9/car_seg_avoid.exe"}
    task_pool_static ={
                 "task10": r"D:/Pytorch_RL_SR/algorithm/AMRL/task10/car_seg_avoid.exe",
                 "task11": r"D:/Pytorch_RL_SR/algorithm/AMRL/task11/car_seg_avoid.exe",
                 "task12": r"D:/Pytorch_RL_SR/algorithm/AMRL/task12/car_seg_avoid.exe",
                 "task13": r"D:/Pytorch_RL_SR/algorithm/AMRL/task13/car_seg_avoid.exe",
                 "task14": r"D:/Pytorch_RL_SR/algorithm/AMRL/task14/car_seg_avoid.exe",
                 "task15": r"D:/Pytorch_RL_SR/algorithm/AMRL/task15/car_seg_avoid.exe",
                 "task16": r"D:/Pytorch_RL_SR/algorithm/AMRL/task16/car_seg_avoid.exe",
                 "task17": r"D:/Pytorch_RL_SR/algorithm/AMRL/task17/car_seg_avoid.exe",
                 "task18": r"D:/Pytorch_RL_SR/algorithm/AMRL/task18/car_seg_avoid.exe",
                 "task19": r"D:/Pytorch_RL_SR/algorithm/AMRL/task19/car_seg_avoid.exe",
                 "task20": r"D:/Pytorch_RL_SR/algorithm/AMRL/task20/car_seg_avoid.exe"}
    task_pool_dynim = {
                 "task21": r"D:/Pytorch_RL_SR/algorithm/AMRL/task21/car_seg_avoid.exe",
                 "task22": r"D:/Pytorch_RL_SR/algorithm/AMRL/task22/car_seg_avoid.exe",
                 "task23": r"D:/Pytorch_RL_SR/algorithm/AMRL/task23/car_seg_avoid.exe",
                 "task24": r"D:/Pytorch_RL_SR/algorithm/AMRL/task24/car_seg_avoid.exe",
                 "task25": r"D:/Pytorch_RL_SR/algorithm/AMRL/task25/car_seg_avoid.exe",
                 "task26": r"D:/Pytorch_RL_SR/algorithm/AMRL/task26/car_seg_avoid.exe",
                 "task27": r"D:/Pytorch_RL_SR/algorithm/AMRL/task27/car_seg_avoid.exe",
                 "task28": r"D:/Pytorch_RL_SR/algorithm/AMRL/task28/car_seg_avoid.exe",
                 "task29": r"D:/Pytorch_RL_SR/algorithm/AMRL/task29/car_seg_avoid.exe",
                 "task30": r"D:/Pytorch_RL_SR/algorithm/AMRL/task30/car_seg_avoid.exe"
                 }
    #目标搜索场景
    targe_scene_task = task_pool_target.keys()
    targe_scene_task_choice = random.sample(targe_scene_task,1)
    # 静态障碍物避障
    static_scene_task = task_pool_static.keys()
    static_scene_task_choice = random.sample(static_scene_task, 1)
    # 动态障碍物避障
    dynamic_scene_task = task_pool_dynim.keys()
    dynamic_scene_task_choice = random.sample(dynamic_scene_task, 1)

    if scene_name == "target_scene":
        select_task = targe_scene_task_choice
        select_task_path = targe_scene_task.get(select_task[0])
    if scene_name == "static_scene":
        select_task = static_scene_task_choice
        select_task_path = task_pool_static.get(select_task[0])
    if scene_name == "dynamic_scene":
        select_task = dynamic_scene_task_choice
        select_task_path = task_pool_dynim.get(select_task[0])
    return select_task_path


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

def env_rest_image_conv(env_reset_state_image):
    obs_arry = np.array(env_reset_state_image)
    obs_tensor = torch.from_numpy(obs_arry)
    obs_tensor_input = obs_tensor.unsqueeze(dim=0)
    changge_obs_state = obs_tensor_input.view(1, 3, 84, 84)
    state = changge_obs_state
    net = CNNNet()
    OUTPUT_obs = net.forward(state)
    out_obs_array = OUTPUT_obs[0]
    out_obs_array = out_obs_array.detach().numpy()
    input_obs_state = torch.from_numpy(out_obs_array)
    state = input_obs_state.unsqueeze(dim=0)
    state = state * 10
    state_image1 = state

    return  state_image1

def ray_trans(state_ray):
    lay = nn.Linear(202, 8)
    # state_ray = state_ray.detach()
    state_ray_tensor = torch.from_numpy(state_ray)
    state_ray1 = lay(state_ray_tensor)
    state_ray1 = state_ray1.detach()
    return state_ray1

def date_jonint(data1,data2):
    data1_list = data1.numpy().tolist()[0]
    data2_list = data2.numpy().tolist()
    data1_list.extend(data2_list)
    total_data = data1_list
    return total_data

def fusion_total_all_sensor(state):
    state_image_front = state[0][0]
    state_image_left = state[1][0]
    state_image_right = state[2][0]
    state_ray = state[3][0]
    state_vector = state[4][0]
    state_image_front_conv = env_rest_image_conv(state_image_front)
    state_image_left_conv = env_rest_image_conv(state_image_left)
    state_image_right_conv = env_rest_image_conv(state_image_right)
    front_image_Wight = Attention(torch.tensor(state_image_front_conv))
    left_image_Wight = Attention(torch.tensor(state_image_left_conv))
    right_image_Wight = Attention(torch.tensor(state_image_right_conv))
    fusion_homogeny_image_state = state_image_front_conv * front_image_Wight + state_image_left_conv * left_image_Wight + state_image_right_conv * right_image_Wight
    ray_conv = ray_trans(state_ray)
    ray_conv_wight = Attention(torch.tensor(ray_conv))
    vector_wight = Attention(torch.tensor(state_vector))
    fusion_hetergeneity_sensor = ray_conv * ray_conv_wight + state_image_front_conv * front_image_Wight + state_vector * vector_wight
    fusion_total_state = date_jonint(fusion_homogeny_image_state, fusion_hetergeneity_sensor)

    return fusion_total_state

def save_final_episode(episdode1):
    os.makedirs(os.path.join('.', 'episode_step'), exist_ok=True)
    episode_step = os.path.join('.', 'episode_step', 'episode.txt')
    with open(episode_step, 'w', encoding='utf-8') as f:
        f.write(str(episdode1))  # 列名
    return episdode1


def Store_Sample(state, action, reward, next_state, done):
        # error = self.loss   #需要传入loss作为值
        error = np.random.rand(1)
        # print("error111111111111111111111111111",error)
        Memory.add(error, (state, action, reward, next_state, done))

def array_tensor(array_numpy):
    pool =[]
    for i in array_numpy:
        i_tensor = torch.tensor(i,dtype=torch.float32)
        pool.append(i_tensor)
    return pool
def array_value(array_value):
    pool =[]
    for i in array_value:
        i_tensor = i.clone().detach()
        i_numpy = i_tensor.numpy()
        pool.append(i_numpy[0])
    return pool
def array_flase(array_value):
    pool =[]
    for i in array_value:
        pool.append(i[0])
    return pool

def per_experence(state,action,reward,fusion_total_state,done,exper_batchsize):
    # ------------添加经验优先回放-------------
    next_state = state
    action = np.expand_dims(action, 0)
    Store_Sample(fusion_total_state, action, reward, next_state, done)  # 存经验
    mini_batch, idxs, is_weights = Memory.sample(exper_batchsize)  # 取经验 mini_batch 经验池，idxs 为下标值，is_weights 权重值
    mini_batch = np.array(mini_batch, dtype=object).transpose()
    states = np.vstack(mini_batch[0])  #   按理说状态应该从一堆状态中进行采样，看代码看是直接是随机数生成的
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_states = np.vstack(mini_batch[3])
    dones = mini_batch[4]
    states1 = array_tensor(states)
    actions1 = array_tensor(actions)
    rewards1 = array_tensor(rewards)
    rewards1 = array_value(rewards1)
    dones = array_flase(dones)
    is_weights1 = is_weights
    is_weights1 = array_tensor(is_weights1)

def sorft_attention(state_dim, action_dim,state_image_list,state_ray_list,state_posi_list,has_continuous_action_space):
    # ---------------使用软注意机制----------------
    attention = Ture
    if attention:
        Attention_net = ActorCritic(state_dim, action_dim, has_continuous_action_space, 0.6)
        Attention_net = Attention_net.MLP
        Image_Wight1 = Attention_net(torch.tensor(state_image_list))  # data 为各个传感器的感知权重
        Ray_Wight2 = Attention_net(torch.tensor(state_ray_list))
        Pos_Wight3 = Attention_net(torch.tensor(state_posi_list))
        # ----------------权重映射-----------------------
        Wight_Dict = dict([('img', Image_Wight1), ('ray', Ray_Wight2), ('pos', Pos_Wight3)])
        # ----------------权重映射-----------------------
        # --------------------比较三个不同权重值---------------
        max_dict_weight = max(zip(Wight_Dict.values(), Wight_Dict.keys()))
        max_type_weight = max_dict_weight[1]  # 输出为字符
        if max_type_weight == 'img':
            state = state_image_list
            # print("以图像为输入")
        if max_type_weight == 'ray':
            state = state_ray_list
            # print("以雷达为输入")
        if max_type_weight == 'pos':
            state = state_posi_list
            # print("以位置信息为输入")
        # -----------------使用软注意力机制-----------

def train():
    # -------------------参数--------------------
    action_dim = 2
    env_name = 'Unitylimocar'
    reword_log = SummaryWriter('./limocar')
    K_epochs = 100  # update policy for K epochs in one PPO update
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.9  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.005  # learning rate for critic network
    random_seed = 0
    max_ep_len = 100
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    print_running_reward = 0
    print_running_episodes = 1
    run_num_pretrained = 0
    current_ep_reward = 0
    directory = "PPO_model"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    # -------------------参数--------------------
    # -----------制作虚实结合的log文件--------------
    show_SR_figure = mlog.run()
    # -----------记录奖励函数曲线-------------------
    reword_log = SummaryWriter('./car')
    # -----------打印unity相关的信息---------------
    logging.basicConfig(level=logging.INFO)
    # -----------获得参数信息----------------------
    parmater = get_args()  # 这个还不太会使用
    # par1 = parmater("--task")
    env_path = outloop_select_task("target_scene")
    # env = UnityWrapper(train_mode=True, base_port=5004, file_name=r"D:\RL_SR\envs\test\car_seg_avoid.exe")
    env = UnityWrapper(train_mode=True, base_port=5004, file_name= env_path)
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    state_dim = obs_shape_list
    print("总的状态维度：", state_dim)  # (前摄像头图像，左摄像头图像，右摄像头图像，射线数据，目标位置和速度)
    print("前摄像头维度：", state_dim[0])
    print("左摄像头维度：", state_dim[1])
    print("右摄像头维度：", state_dim[2])
    print("雷达维度：", state_dim[3])
    print("向量维度：", state_dim[4])

    AMRL_agent = AMRL_Algorithm(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # train makeure task
    for episode in tqdm(range(100)):
        state = env.reset()
        state = fusion_total_all_sensor(state)
        for step in range(2000):
            action = AMRL_agent.selection_action(state)
            action = np.expand_dims(action, 0)
            state, reward, done, _ = env.step(None, action)
            state = fusion_total_all_sensor(state)
            reward = float(reward[0])
            done = bool(done[0])
            AMRL_agent.buffer.rewards.append(reward)
            AMRL_agent.buffer.is_terminals.append(done)
            current_ep_reward += reward
            # update PPO agent
            if episode == 2:
                loss = AMRL_agent.network_update()
                loss2 = loss
            if episode == 2:
                AMRL_agent.decay_action_std(action_std_decay_rate, min_action_std)
            if episode == 1:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 4) * 100  # 取两位有效数字
                reword_log.add_scalar('rewardwithepisode', print_avg_reward,episode)
                # reword_log.add_scalar('loss', loss2, i_episode)
                print("Episode : {}  \t\t Average Reward : {}".format(episode, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0
                # save model weights
            if episode == 1:
                print("saving model at : " + checkpoint_path)
                AMRL_agent.save_network_parm(checkpoint_path)
                print("model saved")
            if done:
                break
            print_running_reward += current_ep_reward / 1000
            print_running_episodes += 1

            save_step_episode = save_final_episode(episode)

    env.close()


def test():
    print("============================================================================================")
    target_search_scene_test_env = r"D:\Pytorch_RL_SR\algorithm\AMRL\task1\car_seg_avoid.exe"
    static_avoid_scene_test_env = r"D:\Pytorch_RL_SR\algorithm\AMRL\task1\car_seg_avoid.exe"
    dynamic_avoid_scene_test_env = r"D:\Pytorch_RL_SR\algorithm\AMRL\task1\car_seg_avoid.exe"
    env_name = "usvcpugridetomove06100848action3"
    has_continuous_action_space = True
    max_ep_len = 1000  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 1000  # total num of testing episodes

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    #####################################################
    logging.basicConfig(level=logging.INFO)
    env = UnityWrapper(train_mode=False, base_port=5004,file_name=static_avoid_scene_test_env)  # 测试三种环境分别是目标搜索，静态障碍物避障和动态障碍物避障
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    # state space dimension
    state_dim = obs_shape_list[0][0][0]
    # state_dim = env.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        # action_dim = env.action_space.shape[0]
        action_dim = c_action_dim
    else:
        action_dim = d_action_dim
        # action_dim = env.action_space.n
    # initialize a PPO agent
    AMRL_agent = AMRL_Algorithm(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # preTrained weights directory
    random_seed = 0  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "PPO_model" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    AMRL_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        state = state[0][0]
        for t in range(1, max_ep_len + 1):
            action = AMRL_agent.select_action(state)
            state, reward, done, _ = env.step(None, np.expand_dims(action, 0))
            state = state[0][0]
            reward = float(reward[0])
            done = bool(done[0])
            ep_reward += reward
            if render:
                # env.render()
                time.sleep(frame_delay)
            if done:
                break
        # clear buffer
        AMRL_agent.buffer.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()
    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")

def image_add_ray_total_state(state_image1,state_ray1):
    state_image_list = state_image1.numpy().tolist()[0]
    state_ray_list = state_ray1.numpy().tolist()
    state_image_list.extend(state_ray_list)
    stte_total_irp = state_image_list
    state = stte_total_irp
    return state

def env_step_image_conv(state_image11):
    obs_arry1 = np.array(state_image11)
    obs_tensor2 = torch.from_numpy(obs_arry1)
    obs_tensor_input2 = obs_tensor2.unsqueeze(dim=0)
    changge_obs_state3 = obs_tensor_input2.view(1, 3, 84, 84)
    state = changge_obs_state3
    net4 = CNNNet()
    OUTPUT_obs5 = net4.forward(state)
    out_obs_array6 = OUTPUT_obs5[0]
    out_obs_array7 = out_obs_array6.detach().numpy()
    input_obs_state8 = torch.from_numpy(out_obs_array7)
    state = input_obs_state8.unsqueeze(dim=0)
    env_step_state_image = state * 10

    return env_step_state_image

def reaL_limocar_test():
    from ros_car import RosCar
    env_name = 'Unitylimocar'
    reword_log = SummaryWriter('./limocar')
    K_epochs = 100  # update policy for K epochs in one PPO update
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.9  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.005  # learning rate for critic network
    random_seed = 0
    max_ep_len = 100
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    run_num_pretrained = 0
    current_ep_reward = 0
    load_model = True
    image_conv = True
    directory = "PPO_model"
    total_test_episodes = 1000
    c_action_dim = 2
    limocar = RosCar()

    checkpoint_path = r"./mode/PPO_Unitylimocar_0_0.pth"

    # True表示需要卷积,False
    if (image_conv):
        state_dim = 16  # 这一步非常重要
    else:
        # 仅有位置信息
        state_dim = 8
    # 确定智能体
    action_dim = 2
    AMRL_agent = AMRL_Algorithm(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # 是否加载模型
    if (load_model):
        AMRL_agent.load_network_parm(checkpoint_path)  # 打开加载权重继续训练
        # i_episode = load_final_episode()
        print("加载模型进行验证实验")
    else:
        print("重新训练")
        i_episode = 0

    time_step = 0

    # while time_step <= max_training_timesteps:
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = limocar.get_rl_obs_list()
        state_image = state[0][0]  # 一张图像被处理成8个数字
        state_ray = state[1][0]  # 404个数据
        #  处理成env.reset 的图像数据
        # --------图像处理-----
        state_image1 = env_rest_image_conv(state_image)
        # ----------雷达处理-------
        state_ray = state_ray[0:202]
        state_ray1 = ray_trans(state_ray)
        # ----------图像和雷达数据合并-------
        state = image_add_ray_total_state(state_image1, state_ray1)

        for t in range(1, max_ep_len + 1):
            # select action with policy #env.reset 产生的state 需要卷积处理
            action = AMRL_agent.selection_action(state)
            action = np.expand_dims(action, 0)
            # state, reward, done, _ = env.step(None, action)
            state, reward, done, _ = limocar.env_step(action)
            state_image11 = state[0][0]  # 一张图像被处理成8个数字
            state_ray11 = state[1][0]  # 404个数据
            # 这是处理env.stp 获得图像数据
            # ----------------处理图像--------------------------------
            env_step_state_image = env_step_image_conv(state_image11)
            # ---------------雷达数据处理----------
            state_ray11 = state_ray[0:202]
            ray_step_state = ray_trans(state_ray11)
            # ----------图像和雷达数据合并-------
            state = image_add_ray_total_state(env_step_state_image, ray_step_state)

    limocar.stop()


if __name__ == "__main__":
    # main()
    train()
