import os
import glob
import time
from datetime import datetime
# from  env  import env
import torch
import numpy as np
import gym
# import roboschool
# import pybullet_envs
from PPO import PPO
import unity_python as upy
#################################### Testing ###################################
def test():
    print("============================================================================================")
    env_name = "usvcpugridetomove06100848action3"
    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving
    
    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1000    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    upy.logging.basicConfig(level=upy.logging.INFO)
    env = upy.UnityWrapper(train_mode=False, base_port=5004)
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
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_model" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()
        state = state[0][0]

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
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
        ppo_agent.buffer.clear()
        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
    env.close()
    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")




if __name__ == '__main__':

    test()
