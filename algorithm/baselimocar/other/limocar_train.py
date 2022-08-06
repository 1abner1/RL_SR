'''
1.pip install mlagents==0.25.0
2.针对mlagents-release 
3.pip install toch gym numpy==1.20.3
4.使用cuda 10.2  pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
5.不能使用cuda 11.1 会出现错误
6.使用mlagents gpu 训练同样采用car这个环境
version = xww  time=20210608
注意：
state_dim= obs_list[0][0][0]
7.使用图像训练时千万注意env.init 返回了state 和 env.step 返回的state 都转化为必须转化为[1,21186](1,84*84*3)
8. 运行环境为unity_env_master
'''
import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
from PPO import PPO
from PPO import ActorCritic
# import PPO
import unity_python as upy
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import pandas as pd
from image_to_conv import CNNNet
# print("111111111111111111111111111")
################################### Training ###################################
def save_final_episode(episdode1):
    os.makedirs(os.path.join('..', 'episode_step'), exist_ok=True)
    episode_step = os.path.join('..', 'episode_step', 'episode.txt')
    with open(episode_step, 'w', encoding='utf-8') as f:
        f.write(str(episdode1))  # 列名
    return episode_step

def load_final_episode():
    with open(r"D:\xwwppounityrollerimage_ray_position_softattention_per_experience\episode_step\episode.txt", 'r') as f:
        data = f.read()
    return int(data)
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


def train():
    print("============================================================================================")
    ####### initialize environment hyperparameters ######
    reword_log = SummaryWriter('./limocar20220728')
    env_name = "limocar_image_ray"
    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100                    # max timesteps in one episode
    max_training_timesteps = int(3e8)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 1        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e3)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e4)  # action_std decay frequency (in num timesteps)
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 100               # update policy for K epochs in one PPO update
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.9            # discount factor
    lr_actor = 0.0003     # learning rate for actor network
    lr_critic = 0.005       # learning rate for critic network
    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################
    print("training environment name : " + env_name)
    upy.logging.basicConfig(level=upy.logging.INFO)
    # env = f'D:\usv\unityexe\UnityEnvironment.exe'  
    env = upy.UnityWrapper(train_mode=True, base_port=5008,no_graphics=False,file_name=r"D:\xwwppounityrollerimage_ray_position_softattention_per_experience\unitycar\car_nvigation.exe")#r"D:\xwwppounityrollerposition\unitycar\car_nvigation.exe")
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    # 状态维度
    state_dim = obs_shape_list[0][0]
    exper_batchsize = 10
    # action space dimension
    if has_continuous_action_space:
        action_dim = c_action_dim
    else:
        action_dim = d_action_dim
    ###################### logging ######################
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder
    directory = "PPO_model"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    image_conv = True  # True表示需要卷积,False
    if (image_conv):
        state_dim = 8  # 这一步非常重要
    else:
        # 仅有位置信息
        state_dim = 8
        # 全部图像作为输入，没有进行任何卷积处理
        # state_dim = state_dim[0]*state_dim[1]*state_dim[2]
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    # 是否加载模型
    load_model = False
    if (load_model):
        ppo_agent.load(checkpoint_path)   # 打开加载权重继续训练
        print("加载上次训练模型继续训练")
        print("已经开始执行")
    else:
        print("重新训练")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 1
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    if (load_model):
        i_episode =load_final_episode()
    else:
        i_episode = 0
    while time_step <= max_training_timesteps:
        state = env.reset()
        state_image = state[0][0]   #一张图像被处理成8个数字
        # print("state_image2222222222",state_image)
        state_ray = state[1][0]     #404个数据
        #state_posi = state[2][0]    # 8 个数字

        if (image_conv):
            #开始图像处理
            obs_arry = np.array(state_image)
            obs_tensor = torch.from_numpy(obs_arry)
            obs_tensor_input = obs_tensor.unsqueeze(dim=0)
            changge_obs_state = obs_tensor_input.view(1,3,84,84)
            state = changge_obs_state
            net = CNNNet()
            OUTPUT_obs = net.forward(state)
            out_obs_array = OUTPUT_obs[0]
            out_obs_array = out_obs_array.detach().numpy()
            input_obs_state = torch.from_numpy(out_obs_array)
            state = input_obs_state.unsqueeze(dim=0)
            state = state*10
            #需要将多种传感器数据进行融合增加这一步
            state_image1 = state
            # print("卷积之后的状态111111111111env.reset产生的",state_image1)
            #结束图像处理
        else:
            state_dim = 8
        #----------------雷达数据处理----------------
        lay = nn.Linear(404, 8)
        # state_ray = state_ray.detach()
        state_ray_tensor = torch.from_numpy(state_ray)
        state_ray1 = lay(state_ray_tensor)
        state_ray1 = state_ray1.detach()
        #------------------雷达数据处理-----------------

        #--------------向量与位置信息------------------------
        state_posi = torch.from_numpy(state_posi)
        state_posi1 = state_posi
        #--------------向量与位置信息-------------------------
        state_image_list = state_image1.numpy().tolist()[0]
        state_ray_list = state_ray1.numpy().tolist()
        state_posi_list = state_posi1.numpy().tolist()
        # print("位置list",state_posi_list)
        # print("图像list",state_image_list)
        # print("雷达list", state_ray_list)
        attention = True
        # ---------------使用软注意机制----------------
        if attention:
            Attention_net = ActorCritic(state_dim, action_dim, has_continuous_action_space, 0.6)
            Attention_net = Attention_net.MLP
            Image_Wight1 = Attention_net(torch.tensor(state_image_list))  # data 为各个传感器的感知权重
            Ray_Wight2 = Attention_net(torch.tensor(state_ray_list))
            Pos_Wight3 = Attention_net(torch.tensor(state_posi_list))
            #----------------权重映射-----------------------
            Wight_Dict= dict([('img',Image_Wight1),('ray',Ray_Wight2),('pos',Pos_Wight3)])
            # ----------------权重映射-----------------------
            #--------------------比较三个不同权重值---------------
            max_dict_weight = max(zip(Wight_Dict.values(),Wight_Dict.keys()))
            max_type_weight = max_dict_weight[1]  #输出为字符
            if max_type_weight =='img':
                state = state_image_list
                # print("以图像为输入")
            if max_type_weight == 'ray':
                state = state_ray_list
                # print("以雷达为输入")
            if max_type_weight == 'pos':
                state = state_posi_list
                # print("以位置信息为输入")

        else:
            state_image_list.extend(state_ray_list)
            state_image_list.extend(state_posi_list)
            # print("合并雷达和图像数据和位置",len(state_image_list))
            stte_total_irp = state_image_list
            state = stte_total_irp
        # -----------------使用软注意力机制-----------
        state1 = state
        current_ep_reward = 0
        for t in range(1, max_ep_len+1):
            # select action with policy #env.reset 产生的state 需要卷积处理
            # print("state11111111111111111222222222222222",state)
            action = ppo_agent.select_action(state)
            # print("输出的动作：",action)
            state, reward, done, _ = env.step(None, np.expand_dims(action, 0))   #下一时刻的状态

            #env.step也需要卷积和并处理1111111111111111111111111111111111111111处理
            # print("state4444444444444444444444", state)
            state_image11 = state[0][0]  # 一张图像被处理成8个数字
            # print("state_image2222222222",state_image)
            state_ray11 = state[1][0]  # 404个数据
            state_posi11 = state[2][0]  # 8 个数字
            # env.step也需要卷积和并处理1111111111111111111111111111111111111111处理



            if (image_conv):
                #1111111111111111111111111111111111111111
                obs_arry1 = np.array(state_image11)
                obs_tensor2 = torch.from_numpy(obs_arry1)
                obs_tensor_input2 = obs_tensor2.unsqueeze(dim=0)
                changge_obs_state3 = obs_tensor_input2.view(1, 3,84,84)
                state = changge_obs_state3
                from image_to_conv import CNNNet
                net4 = CNNNet()
                OUTPUT_obs5 = net4.forward(state)
                out_obs_array6 = OUTPUT_obs5[0]
                out_obs_array7 = out_obs_array6.detach().numpy()
                input_obs_state8 = torch.from_numpy(out_obs_array7)
                state = input_obs_state8.unsqueeze(dim=0)
                state = state*10
                # print("state步骤env.step产生的",state)
                #1111111111111111111111111111111111111
                # # #测试
                # state = torch.rand(1, 8)
                # print("state步骤env.step产生的", state)
            else:
                #原来的
                # obs_arry = np.array(state)
                # obs_tensor = torch.from_numpy(obs_arry)
                # changge_obs_state = obs_tensor.view(1, 84 * 84 * 3)
                # state = changge_obs_state
                # print("state11111step产生的",state)
                #原来的

                # 仅仅只有位置信息
                pass
            #111111111111111111111111111111111111111111111111111
            # 雷达数据处理
            lay = nn.Linear(404, 8)
            # state_ray = state_ray.detach()
            state_ray_tensor = torch.from_numpy(state_ray11)
            state_ray1 = lay(state_ray_tensor)
            state_ray1 = state_ray1.detach()
            # 雷达数据处理

            # 向量与位置信息
            state_posi = torch.from_numpy(state_posi11)
            state_posi1 = state_posi
            # 向量与位置信息

            # print("111111111111111111",type(state_ray1.numpy().tolist()))
            state_image_list = state_image1.numpy().tolist()[0]
            state_ray_list = state_ray1.numpy().tolist()
            state_posi_list = state_posi1.numpy().tolist()
            # print("位置list", state_posi_list)
            # print("图像list", state_image_list)
            # print("雷达list", state_ray_list)

            # ---------------使用软注意机制----------------
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
            else:
                state_image_list.extend(state_ray_list)
                state_image_list.extend(state_posi_list)
                # print("合并雷达和图像数据和位置", len(state_image_list))
                stte_total_irp = state_image_list
                state = stte_total_irp

            # ------------添加经验优先回放-------------
            next_state = state
            action = np.expand_dims(action, 0)
            ppo_agent.Store_Sample(state1, action, reward, next_state, done)  # 存经验
            mini_batch, idxs, is_weights = ppo_agent.memory.sample(exper_batchsize)  # 取经验 mini_batch 经验池，idxs 为下标值，is_weights 权重值

            mini_batch = np.array(mini_batch,dtype=object).transpose()
            states = np.vstack(mini_batch[0])  # 按理说状态应该从一堆状态中进行采样，看代码看是直接是随机数生成的
            # print("经验池中获得真实场景数据11111111111111111111111",states)
            actions = list(mini_batch[1])
            # print("经验池中获得动作11111111111111111111111", actions)
            rewards = list(mini_batch[2])
            # print("经验池中获得奖励11111111111111111111111", rewards)
            next_states = np.vstack(mini_batch[3])
            dones = mini_batch[4]
            # print("经验池中获得奖励11111111111111111111111",dones)
            # dones = dones.astype(int)
            states1 = array_tensor(states)
            actions1 = array_tensor(actions)
            # print("actions为4444444444444444444444444444444444444",actions1)
            # print("rewards为4444444444444444444444444444444444444",rewards)
            rewards1 = array_tensor(rewards)
            rewards1 =array_value(rewards1)
            # print("actions为4444444444444444444444444444444444444",rewards1)
            # dones1 = array_tensor(dones)
            dones = array_flase(dones)
            # print("actions为4444444444444444444444444444444444444",dones)
            is_weights1 = is_weights
            is_weights1 = array_tensor(is_weights1)
            # print("states12333333333333333333333333333333",is_weights1)
            # ------------添加经验优先回放-------------
            # 111111111111111111111111111111111111111111111111111
            reward = float(reward[0])
            done = bool(done[0])
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward
            loss2 = torch.tensor(0)
            # update PPO agent
            if time_step % update_timestep == 0:
                loss,loss3 = ppo_agent.update(states1,actions1,rewards1,is_weights1,dones)   #所有的更新都在这将采样中的状态，动作，权重值传入进去
                # print("loss",loss)
                loss2 = loss
                # print("loss211111111111",loss2)
                # print("loss35555555555555555555555555555555555",loss3.numpy())
                # -----根据loss 更新经验池--------------------------------------
                errors = loss3.numpy()
                # print("errors12335555555555555555555555555555555555", errors)
                # errors = errors.append(loss2)  #按理说这个只有一个值呀,需要把50 的loss应该存储再一个池子中
                # print("errors",errors)
                for i in range(exper_batchsize):
                    idx = idxs[i]
                    # print("idx111111111111111", idx)
                    ppo_agent.memory.update(idx, errors[i])
                # -----根据loss 更新经验池--------------------------------------

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                # print("1111111111111111111111111111111111111")
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 4)*100 #取两位有效数字
                reword_log.add_scalar('rewardwithepisode',print_avg_reward,i_episode)
                # print("输出的ppo_agentlos11111111",loss1)
                reword_log.add_scalar('loss', loss2,i_episode)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t totallos:{}".format(i_episode, time_step, print_avg_reward,loss2))
                print_running_reward = 0
                print_running_episodes = 0
                # self.final_episode = i_episode

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward/1000
        print_running_episodes += 1

        log_running_reward += current_ep_reward/1000
        log_running_episodes += 1

        i_episode += 1
        save_step_episode= save_final_episode(i_episode)
    log_f.close()
    print("执行关闭环境之前的动作")
    env.close()
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time  : ", end_time - start_time)

if __name__ == '__main__':
    train()
  
    
    
    
    
    
    
    
