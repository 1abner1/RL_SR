'''
1.目前支持离散动作的算法有10 中TD3,PPO,ddpg,sac,trpo,VPG,NPG,ASAC,tac,atac
2.目前只是可以跑通，训练效果还没有具体验证，最后可能只支持unity 中离散的动作
3.不能打开tensorboard 曲线图
'''
import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
import logging
import unity_wrapper   # 已经把这个unity_wrapper 这个文件已经放在了run_unity.py 这个文件夹下了，才可以使用
# from datetime import datetime
# from env_wrapper.unity_wrapper import unity_wrapper
# import env_wrapper.unity_wrapper as unity_wrapper
from torch.utils.tensorboard import SummaryWriter
# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='unity_car_ppo1', 
                    help='pendulum environment') # 'Pendulum-v0'
<<<<<<< HEAD
parser.add_argument('--algo', type=str, default='ppo_class',
=======
parser.add_argument('--algo', type=str, default='ppo', 
>>>>>>> master
                    help='select an algorithm among vpg, npg, trpo, ppo, ddpg, td3, sac, asac, tac, atac')
parser.add_argument('--phase', type=str, default='test',  #'train' 'test'
                    help='choose between training phase and testing phase')
parser.add_argument('--render', action='store_true', default=False,
                    help='if you want to render, set this to True')
parser.add_argument('--load', type=str, default=True,
                    help='copy & paste the saved model name, and load it') #True #False
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=1000000000, 
                    help='iterations to run and train agent')
parser.add_argument('--eval_per_train', type=int, default=10, 
                    help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=100,
                    help='max episode step')
parser.add_argument('--threshold_return', type=int, default=-230,
                    help='solved requirement for success in given environment')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=0)  # 1表示使用GPU 进行训练
# parser.add_argument('--debug', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'vpg':
    from agents.vpg import Agent
elif args.algo == 'npg':
    from agents.trpo import Agent
elif args.algo == 'trpo':
    from agents.trpo import Agent
elif args.algo == 'ppo':
<<<<<<< HEAD
    print("1111111111111111执行ppo算法1111111111111111111111111111111111111111111")
    from agents.ppo import Agent   #首先运行这一步
elif args.algo == 'ppo_class':
    from agents.ppo_class import Agent  # ppo 整合算法
=======
    print("11111111111111111111111111111111111111111111111111111111111")
    from agents.ppo import Agent   #首先运行这一步
>>>>>>> master
elif args.algo == 'ddpg':
    from agents.ddpg import Agent
elif args.algo == 'td3':
    from agents.td3 import Agent
elif args.algo == 'sac':
    from agents.sac import Agent
elif args.algo == 'asac': # Automating entropy adjustment on SAC
    from agents.sac import Agent
elif args.algo == 'tac': 
    from agents.sac import Agent
elif args.algo == 'atac': # Automating entropy adjustment on TAC
    from agents.sac import Agent

<<<<<<< HEAD
class run_unity_rl():
    def __init__(self):
        self.real_to_agent_obs = []
    def main(slef,visobs):
        """Main."""
        # Initialize environment
        self.real_to_agent_obs = visobs
        print("开始运行11111111111111111111111111111111111111111111111111")
        logging.basicConfig(level=logging.INFO)
        env = unity_wrapper.UnityWrapper(train_mode =True,base_port = 5004)
        obs_shapes, discrete_action_size, continuous_action_size=env.init()
        # obs_dim = obs_shapes[1][0]
        # obs_dim = int(obs_shapes[0][0]) * int(obs_shapes[0][1]) * int(obs_shapes[0][2])
        obs_dim = 8
        act_dim = discrete_action_size
        continuous_action_size = None
        act_limit = 2
        # start_time = datetime.now().replace(microsecond=0)
        # print("Started training at (GMT) : ", start_time)
        print('---------------------------------------')
        print('Environment:', args.env)
        print('Algorithm:', args.algo)
        print('State dimension:', obs_dim)
        print('Action dimension:', act_dim)
        print('Action limit:', act_limit)
        print('---------------------------------------')

        # Set a random seed
        # env.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)

        # Create an agent
        if args.algo == 'ddpg' or args.algo == 'td3':
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit)
        elif args.algo == 'sac':
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit,
                          alpha=0.5)
        elif args.algo == 'asac':
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit,
                          automatic_entropy_tuning=True)
        elif args.algo == 'tac':
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit,
                          alpha=0.5,)
                        #   log_type='log-q',
                        #  entropic_index=1.2)
        elif args.algo == 'atac':
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit,
                        #   log_type='log-q',
                        #   entropic_index=1.2,
                          automatic_entropy_tuning=True)
        elif args.algo == 'ppo_class':
            # 重新定义一个可以执行ppo
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit,real_obs)

        else: # vpg, npg, trpo, ppo
            agent = Agent(env, args, device, obs_dim, act_dim, act_limit)




        # If we have a saved model, load it
        # if args.load is not None:
        if args.load:
            new_pth_file =os.listdir(r"D:\Pytorch_RL _sort\algorithm\deep_rl\save_model")
            args.load = new_pth_file[-1]
            print("模型加载成功",args.load)  #模型加载成功是否可以继续训练？
            pretrained_model_path = os.path.join('./save_model/' + str(args.load))
            pretrained_model = torch.load(pretrained_model_path, map_location=device)
            agent.policy.load_state_dict(pretrained_model)

        # Create a SummaryWriter object by TensorBoard
        if args.tensorboard:
        # if args.tensorboard and args.load is None:
            dir_name = 'logs/' + args.env + '/' \
                               + args.algo \
                               + '_s_' + str(args.seed) \
                               + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir=dir_name)

        start_time = time.time()

        train_num_steps = 0
        train_sum_returns = 0.
        train_num_episodes = 0

        # Main loop
        for i in range(args.iterations):
            # Perform the training phase, during which the agent learns
            if args.phase == 'train':
                agent.eval_mode = False

                # Run one episode
                train_step_length, train_episode_return,action_ros = agent.run(args.max_step)

                train_num_steps += train_step_length
                train_sum_returns += train_episode_return
                train_num_episodes += 1

                train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

                # Log experiment result for training episodes
                if args.tensorboard:
                # if args.tensorboard and args.load is None:
                    writer.add_scalar('Train/AverageReturns', train_average_return, i)
                    writer.add_scalar('Train/EpisodeReturns', train_episode_return, i)
                    if args.algo == 'asac' or args.algo == 'atac':
                        writer.add_scalar('Train/Alpha', agent.alpha, i)

            # Perform the evaluation phase -- no learning
            if (i + 1) % args.eval_per_train == 0:
                eval_sum_returns = 0.
                eval_num_episodes = 0
                agent.eval_mode = False

                for _ in range(10):
                    # Run one episode
                    eval_step_length, eval_episode_return,action_ros = agent.run(args.max_step)

                    eval_sum_returns += eval_episode_return
                    eval_num_episodes += 1

                eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0
                # end_time = datetime.now().replace(microsecond=0)
                # Log experiment result for evaluation episodes
                if args.tensorboard:
                # if args.tensorboard and args.load is None:
                    writer.add_scalar('Eval/AverageReturns', eval_average_return, i)
                    writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, i)

                if args.phase == 'train':
                    train_average_return[0] = round(train_average_return[0],3)
                    train_average_return[0] = str(train_average_return[0])
                    print(f"training_Episodes:{train_num_episodes}|Timestep:{int(time.time() - start_time)}|Average Reward:{train_average_return[0]:.4f}")

                    # Save the trained model
                    if eval_average_return >= args.threshold_return:
                        if not os.path.exists('./save_model'):
                            os.mkdir('./save_model')

                        ckpt_path = os.path.join('./save_model/' + args.env + '_' + args.algo \
                                                                            + '_s_' + str(args.seed) \
                                                                            + '_i_' + str(i + 1) \
                                                                            + '_tr_' + str(round(train_episode_return[0], 2)) \
                                                                            +  '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")\
                                                                            + '_er_' + str(round(eval_episode_return[0], 2)) + '.pth')

                        torch.save(agent.policy.state_dict(), ckpt_path)
                elif args.phase == 'test':
                    mean_reard1 = eval_average_return[0]
                    print(f"testing_Episodes:{train_num_episodes}|Timestep:{int(time.time() - start_time)}|Average Reward:{mean_reard1:.4f}")
        real_action = [0]
        real_action = action_ros
        return real_action
# if __name__ == "__main__":
#     main()
=======
def main():
    """Main."""
    # Initialize environment
    print("开始运行11111111111111111111111111111111111111111111111111")
    logging.basicConfig(level=logging.INFO)
    env = unity_wrapper.UnityWrapper(train_mode =True,base_port = 5004)
    obs_shapes, discrete_action_size, continuous_action_size=env.init()
    # obs_dim = obs_shapes[1][0]
    # obs_dim = int(obs_shapes[0][0]) * int(obs_shapes[0][1]) * int(obs_shapes[0][2])
    obs_dim = 8
    act_dim = discrete_action_size
    continuous_action_size = None
    act_limit = 2
    # start_time = datetime.now().replace(microsecond=0)
    # print("Started training at (GMT) : ", start_time)
    print('---------------------------------------')
    print('Environment:', args.env)
    print('Algorithm:', args.algo)
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)
    print('Action limit:', act_limit)
    print('---------------------------------------')

    # Set a random seed
    # env.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # Create an agent
    if args.algo == 'ddpg' or args.algo == 'td3':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit)
    elif args.algo == 'sac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      alpha=0.5)
    elif args.algo == 'asac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      automatic_entropy_tuning=True)
    elif args.algo == 'tac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      alpha=0.5,)
                    #   log_type='log-q', 
                    #  entropic_index=1.2)
    elif args.algo == 'atac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                    #   log_type='log-q', 
                    #   entropic_index=1.2, 
                      automatic_entropy_tuning=True)
    else: # vpg, npg, trpo, ppo
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit)

    # If we have a saved model, load it
    # if args.load is not None:
    if args.load:
        new_pth_file =os.listdir(r"D:\Pytorch_RL _sort\algorithm\deep_rl\save_model")
        args.load = new_pth_file[-1]
        print("模型加载成功",args.load)  #模型加载成功是否可以继续训练？
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        agent.policy.load_state_dict(pretrained_model)

    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard:
    # if args.tensorboard and args.load is None:
        dir_name = 'logs/' + args.env + '/' \
                           + args.algo \
                           + '_s_' + str(args.seed) \
                           + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Main loop
    for i in range(args.iterations):
        # Perform the training phase, during which the agent learns
        if args.phase == 'train':
            agent.eval_mode = False
            
            # Run one episode
            train_step_length, train_episode_return = agent.run(args.max_step)
            
            train_num_steps += train_step_length
            train_sum_returns += train_episode_return
            train_num_episodes += 1

            train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

            # Log experiment result for training episodes
            if args.tensorboard:
            # if args.tensorboard and args.load is None:
                writer.add_scalar('Train/AverageReturns', train_average_return, i)
                writer.add_scalar('Train/EpisodeReturns', train_episode_return, i)
                if args.algo == 'asac' or args.algo == 'atac':
                    writer.add_scalar('Train/Alpha', agent.alpha, i)

        # Perform the evaluation phase -- no learning
        if (i + 1) % args.eval_per_train == 0:
            eval_sum_returns = 0.
            eval_num_episodes = 0
            agent.eval_mode = False

            for _ in range(10):
                # Run one episode
                eval_step_length, eval_episode_return = agent.run(args.max_step)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

            eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0
            # end_time = datetime.now().replace(microsecond=0)
            # Log experiment result for evaluation episodes
            if args.tensorboard:
            # if args.tensorboard and args.load is None:
                writer.add_scalar('Eval/AverageReturns', eval_average_return, i)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, i)

            if args.phase == 'train':
                train_average_return[0] = round(train_average_return[0],3)
                train_average_return[0] = str(train_average_return[0])
                print(f"training_Episodes:{train_num_episodes}|Timestep:{int(time.time() - start_time)}|Average Reward:{train_average_return[0]:.4f}")
    
                # Save the trained model
                if eval_average_return >= args.threshold_return:
                    if not os.path.exists('./save_model'):
                        os.mkdir('./save_model')
                    
                    ckpt_path = os.path.join('./save_model/' + args.env + '_' + args.algo \
                                                                        + '_s_' + str(args.seed) \
                                                                        + '_i_' + str(i + 1) \
                                                                        + '_tr_' + str(round(train_episode_return[0], 2)) \
                                                                        +  '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")\
                                                                        + '_er_' + str(round(eval_episode_return[0], 2)) + '.pth')
                    
                    torch.save(agent.policy.state_dict(), ckpt_path)
            elif args.phase == 'test':
                mean_reard1 = eval_average_return[0]
                print(f"testing_Episodes:{train_num_episodes}|Timestep:{int(time.time() - start_time)}|Average Reward:{mean_reard1:.4f}")

if __name__ == "__main__":
    main()
>>>>>>> master
