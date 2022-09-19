#安装虚拟环境要求仅requirements.txt
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from unity_API import UnityWrapper
from torch.distributions import MultivariateNormal
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
import logging


transformer = transforms.Compose([transforms.Resize((84, 84)),    #resize 调整图片大小
                                  # transforms.RandomHorizontalFlip(), # 水平反转
                                  transforms.ToTensor(),  # 0-255 to 0-1 归一化处理
                                  # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  #归一化
                                  ])

device = torch.device('cuda:0') #'cuda:0'

def load_final_episode():
    with open(".tem/episode.txt", 'r') as f:
        data = f.read()
    return int(data)

def save_final_episode(episdode1):
    os.makedirs(os.path.join('.', 'episode_step'), exist_ok=True)
    episode_step = os.path.join('.', 'episode_step', 'episode.txt')
    with open(episode_step, 'w', encoding='utf-8') as f:
        f.write(str(episdode1))  # 列名
    return episdode1

def env_rest_image_conv(env_reset_state_image):
    obs_arry = np.array(env_reset_state_image)
    obs_tensor = torch.from_numpy(obs_arry)
    obs_tensor_input = obs_tensor.unsqueeze(dim=0)
    changge_obs_state = obs_tensor_input.view(1, 3, 84, 84)
    state = changge_obs_state
    net = CNNNet()
    net = net.to(device)
    # print("卷积状态net111111111111111111111",state)
    state = state.to(device)
    OUTPUT_obs = net.forward(state)
    out_obs_array = OUTPUT_obs[0]
    out_obs_array = out_obs_array.cpu()
    out_obs_array = out_obs_array.detach().numpy()
    input_obs_state = torch.from_numpy(out_obs_array)
    state = input_obs_state.unsqueeze(dim=0)
    state = state * 10
    state_image1 = state

    return  state_image1

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

def ray_trans(state_ray):
    lay = nn.Linear(202, 64)
    # state_ray = state_ray.detach()
    state_ray_tensor = torch.from_numpy(state_ray)
    lay = lay.to(device)
    # print("state_ray_tensor类型",type(state_ray_tensor))
    state_ray_tensor = state_ray_tensor.to(device)
    state_ray1 = lay(state_ray_tensor)
    state_ray1 = state_ray1.detach()
    # print("state_ray1类型",type(state_ray1))
    state_ray1 = state_ray1.cpu()
    return state_ray1

def image_add_ray_total_state(state_image1,state_ray1):
    state_image_list = state_image1.numpy().tolist()[0]
    state_ray_list = state_ray1.numpy().tolist()
    state_image_list.extend(state_ray_list)
    stte_total_irp = state_image_list
    state = stte_total_irp

    return state


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
        # state_dim = torch.tensor(state_dim).to(device)
        # action_dim = torch.tensor(action_dim).to(device)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def Actor_policy(self, state):
        self.actor = self.actor.to(device)  #模型也需要放在gpu中
        action_mean = self.actor(state)  # 这一部有问题
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
        self.critic = self.critic.to(device)  # 更新critic 网络到gpu 中
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

class PPO_Algorithm():

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
            # print("state55555555555555类型",type(state))  # 这个state 是在gpu 内存中
            action, action_logprob = self.policy_value.Actor_policy(state) #关键点
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
            self.MseLoss = self.MseLoss.to(device)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,rewards) - 0.01 * dist_entropy  # dist_entropy 为多元高斯分布的信息熵(传入动作的信息熵) 最后一项表示的增加探索项。
            self.loss = loss
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # loss2.mean().backward()
            self.optimizer.step()
            loss1 = loss.mean()
            # loss2 = loss2.mean()
        # Copy new weights into old policy
        self.policy_value_old.load_state_dict(self.policy_value.state_dict())
        # clear buffer
        self.buffer.clear()
        print("loss16666666666666666666",loss1)
        # print("loss26666666666666666666", loss2)

        return loss1

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


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 定义第一个卷积层 84 *84 *3,out_channels=16 # 通道数越大，计算时间越长
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1,padding=1,bias=False)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1,padding=1)
        # 卷积之后的图像大小out_channels*(84-kernel_size=2+2*padding=1)/stride=1 + 1 ;(84-2 + 2*1)
        # 定义第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        # (85 -2 +2*1)/1 + 1=86  输出通道数 16*86*86
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1,padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1,padding=1)
        #(86-2+2*1)/1  +1 ;3*83 *83
        # 定义第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        #16*88*88
        # 定义第一个全连接层
        self.fc1 = nn.Linear(3*84*84, 512)  #4096
        # 定义第二个全连接层
        self.fc2 = nn.Linear(512, 64)    #4096

    def forward(self, x):
        e1 = self.conv1(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 3*84*84)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x


def train():
    env_name = 'Unitylimocar'
    reword_log = SummaryWriter('./limocar/train_zzy_env')
    K_epochs = 2000  # update policy for K epochs in one PPO update
    eps_clip = 0.3  # clip parameter for PPO
    gamma = 0.98  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.005  # learning rate for critic network
    random_seed = 0
    max_ep_len = 1000
    log_freq = max_ep_len * 2
    print_freq = max_ep_len * 1
    update_timestep = max_ep_len * 4
    save_model_freq = int(1e3)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e4)
    max_training_timesteps = int(3e8)
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 1
    run_num_pretrained = 0
    current_ep_reward = 0
    load_model = False
    image_conv = True
    directory = "PPO_model"

    logging.basicConfig(level=logging.INFO)
    env = UnityWrapper(train_mode=True, base_port=5011,file_name=r"D:\zzy_env_ray_position\RLEnvironments.exe")
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    # 状态维度
    state_dim = obs_shape_list[0][0]
    action_dim = c_action_dim
    print("离散动作值",d_action_dim)

    #### change this to prevent overwriting weights in same env_name folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = directory + "PPO2_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    state_dim = 128  # 这一步非常重要

    # 确定智能体
    ppo_agent = PPO_Algorithm(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,action_std)

    # 是否加载模型
    if (load_model):
        ppo_agent.load_network_parm(checkpoint_path)  # 打开加载权重继续训练
        i_episode = load_final_episode()
        print("加载上次训练模型继续训练")
    else:
        print("重新训练")
        i_episode = 0

    start_time = datetime.now().replace(microsecond=0)
    time_step = 0

    while time_step <= max_training_timesteps:
        state = env.reset()
        # print("statereset类型",type(state))
        state_image = state[0][0]#一张图像被处理成8个数字
        state_ray = state[1][0]     #404个数据
        #  处理成env.reset 的图像数据
        #--------图像处理-----
        state_image1 = env_rest_image_conv(state_image)
        #----------雷达处理-------
        state_ray1 = ray_trans(state_ray)
        # ----------图像和雷达数据合并-------
        state = image_add_ray_total_state(state_image1,state_ray1)

        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            # select action with policy #env.reset 产生的state 需要卷积处理
            # print("state类型",type(state))
            # state = torch.tensor(state)
            # state = state.to(device)
            # print("state1类型", type(state))
            action = ppo_agent.selection_action(state)  # 状态传入到了gpu
            action = np.expand_dims(action, 0)
            state, reward, done, _ = env.step(None, action)
            # print("reward",reward)
            state_image11 = state[0][0]  # 一张图像被处理成8个数字
            state_ray11 = state[1][0]  # 404个数据
            # print("reward",reward)
            # 这是处理env.stp 获得图像数据
            # ----------------处理图像--------------------------------
            env_step_state_image = env_step_image_conv(state_image11)
            # ---------------雷达数据处理----------
            ray_step_state = ray_trans(state_ray11)
            # ----------图像和雷达数据合并-------
            state = image_add_ray_total_state(env_step_state_image, ray_step_state)

            # print("合并数据类型",type(state))
            reward = float(reward[0])
            done = bool(done[0])
            # print("reward111111111111111",reward)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward

            loss2 = torch.tensor(0)
            # update PPO agent
            # print("time_step",time_step)
            if time_step % update_timestep == 0:
                loss = ppo_agent.network_update()
                # loss2 = torch.tensor(loss)
                print("更新参数loss",loss)
                reword_log.add_scalar('loss', loss, i_episode)

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # save model weights
            if time_step % save_model_freq == 0:
                print("saving model at : " + checkpoint_path)
                ppo_agent.save_network_parm(checkpoint_path)
                print("model saved")
            if done:
                break
            print_running_reward += current_ep_reward / 1000
            print_running_episodes += 1
        print("Episode:{} Average Reward:{}".format(i_episode, current_ep_reward))

        reword_log.add_scalar('reward_episode',current_ep_reward,i_episode)

        reword_log.add_scalar('rewardwithepisode', i_episode, current_ep_reward)

        i_episode += 1
        save_step_episode = save_final_episode(i_episode)
        # print("执行到第",save_step_episode)
    env.close()
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time : ", end_time - start_time)

def test():
    env_name = 'Unitylimocar'
    reword_log = SummaryWriter('./limocar')
    K_epochs = 100  # update policy for K epochs in one PPO update
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.98  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.005  # learning rate for critic network
    random_seed = 0
    max_ep_len = 1000
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    run_num_pretrained = 0
    current_ep_reward = 0
    load_model = True
    image_conv = True
    directory = "PPO_model"
    total_test_episodes = 1000

    logging.basicConfig(level=logging.INFO)
    env = UnityWrapper(train_mode=False, base_port=50011,file_name=r"D:\RL_SR\envs\limocar_cxz\AURP.exe")
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    # 状态维度
    state_dim = obs_shape_list[0][0]
    action_dim = c_action_dim
    #### change this to prevent overwriting weights in same env_name folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = directory + "PPO1_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    state_dim = 128  # 这一步非常重要

    # 确定智能体
    ppo_agent = PPO_Algorithm(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # 是否加载模型
    if (load_model):
        ppo_agent.load_network_parm(checkpoint_path)  # 打开加载权重继续训练
        # i_episode = load_final_episode()
        print("加载模型进行验证实验")
    else:
        print("重新训练")
        i_episode = 0

    start_time = datetime.now().replace(microsecond=0)
    time_step = 0


    current_ep_reward = 0

    # while time_step <= max_training_timesteps:
    for ep in range(1, total_test_episodes + 1):
        current_ep_reward = 0
        ep_reward = 0
        state = env.reset()
        state_image = state[0][0]  # 一张图像被处理成8个数字
        state_ray = state[1][0]  # 404个数据
        #  处理成env.reset 的图像数据
        # --------图像处理-----
        state_image1 = env_rest_image_conv(state_image)
        # ----------雷达处理-------
        state_ray1 = ray_trans(state_ray)
        # ----------图像和雷达数据合并-------
        state = image_add_ray_total_state(state_image1, state_ray1)

        for t in range(1, max_ep_len + 1):
            # select action with policy #env.reset 产生的state 需要卷积处理
            action = ppo_agent.selection_action(state)
            action = np.expand_dims(action, 0)
            state, reward, done, _ = env.step(None, action)
            # print("reward",reward)
            state_image11 = state[0][0]  # 一张图像被处理成8个数字
            state_ray11 = state[1][0]  # 404个数据
            # 这是处理env.stp 获得图像数据
            # ----------------处理图像--------------------------------
            env_step_state_image = env_step_image_conv(state_image11)
            # ---------------雷达数据处理----------
            ray_step_state = ray_trans(state_ray11)
            # ----------图像和雷达数据合并-------
            state = image_add_ray_total_state(env_step_state_image, ray_step_state)

            reward = float(reward[0])
            done = bool(done[0])

            # saving reward and is_terminals
            ppo_agent.buffer.clear()
            time_step += 1
            current_ep_reward += reward
        print("Episode:{}\t Average Reward:{}".format(ep, current_ep_reward))

    env.close()

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
    ppo_agent = PPO_Algorithm(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # 是否加载模型
    if (load_model):
        ppo_agent.load_network_parm(checkpoint_path)  # 打开加载权重继续训练
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
            action = ppo_agent.selection_action(state)
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


if __name__ == '__main__':
    train()
    # test()
    # reaL_limocar_test()limocar_test()
    print("end")
