"""
Deep Q-Network Q(a, s)
-----------------------
TD Learning, Off-Policy, e-Greedy Exploration (GLIE).
Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
delta_w = R + lambda * Q(newS, newA)
See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
Reference
----------
original paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327
Note: Policy Network has been proved to be better than Q-Learning, see tutorial_atari_pong.py
Environment
-----------
Prerequisites
--------------
years=20200116
tensorflow>=2.2.0
tensorlayer=2.2.3
numpy =1.19.5
-------
Notes
--------------
1.unity_wrappy_zzy is requirement relase 12 -V=0.23.0
2.Python DQN_image  input in
3.use tensorlayer to buid network
"""
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import unity_wrapper_zzy as uw
# add arguments in command  --train/test
parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller.')  # 创建对象,使用argparse 函数
parser.add_argument('--train', dest='train',
                    action='store_true', default=True)  # 增加参数
parser.add_argument('--test', dest='test',
                    action='store_true', default=False)  # 增加参数
args = parser.parse_args()  # 解析参数
# tl.logging.set_verbosity(tl.logging.INFO)
#####################  hyper parameters  ####################
env_id = 'carfindobject'
alg_name = 'DQN'
lambd = .95  # decay factor
e = 0.0015  # e-Greedy Exploration, the larger the more random
num_episodes = 35000
min_step = 200
# replay_size=100
# BATCH_SIZE=32
# batch_size=32
##################### DQN ##########################
# Define Q-network q(a,s) that ouput the rewards of 5 actions(forward,backe,left,right,stop, discreate) by given state, i.e. Action-Value Function.
def get_model(inputs_shape):
    visual_inputs = tl.layers.Input([None, *inputs_shape], tf.float32, 'state')
    visual_embedding1 = tl.layers.Conv2dLayer(shape=(3, 3, 3, 32), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_1')(visual_inputs)
    visual_embedding2 = tl.layers.Conv2dLayer(shape=(3, 3, 32, 64), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_2')(visual_embedding1)
    # shape设置卷积核的形状，前两维是filter的大小，第三维表示前一层的通道数，也即每个filter的通道数，第四维表示filter的数量(3*3 长宽，3 通道，32 个卷积核)
    # 上面这一部分表示的是对图像进行处理的过程
    inputs = tl.layers.Flatten()(visual_embedding2)
    layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
    layer = tl.layers.Dense(64, tf.nn.relu)(layer)
    v = tl.layers.Dense(5)(layer)  # 5 个输入动作
    return tl.models.Model(inputs=visual_inputs, outputs=v, name="Q-Network")
    # 创建经验池，如何使用呢？
# def experience_pool(self,_obs_list, reward, done, max_step):
#     self.replay_buffer.append((_obs_list,reward,done,max_step))
#     if len(self.replay_buffer)>replay_size:
#         self.replay_buffer.popleft()
#     if len(self.replay_buffer) > BATCH_SIZE:
#       self.get_model()
def save_ckpt(model):  # save trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model_2021_1_22_16_46.hdf5'), model)


def load_ckpt(model):  # load trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    # tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model_19_12.hdf5'), model)
    tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'dqn_model_2021_1_22_16_46.hdf5'), model)


if __name__ == '__main__':
    LogDir = os.path.join("logs123/CarFindObject1_20211007")
    # LogDir = os.path.join("logs123/"+enValueError: Can't convert Python sequence with mixed types to Tensor.v_id)
    LogDir = os.path.join("logs123/CarFindObject1_20210409")
    # LogDir = os.path.join("logs123/"+env_id)
    uw.logging.basicConfig(level=uw.logging.INFO)
    env = uw.UnityWrapper(train_mode=False, base_port=5004)
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    #print("obs_shape_list[0]:",obs_shape_list[0])
    qnetwork = get_model(obs_shape_list[0][0])
    qnetwork = get_model(obs_shape_list[0])
    qnetwork.train()
    train_weights = qnetwork.trainable_weights
    optimizer = tf.optimizers.SGD(learning_rate=0.01)   # 随机策略梯度函数
    t0 = time.time()
    if args.train:
        summary_writer = tf.summary.create_file_writer(LogDir)   # record tensorboard file evenfile
        all_episode_reward = []
        for i in range(num_episodes):
            # Reset environment and get first new observation
            # observation is state, 即一张图片,48*64*3 的数据量
            obs_list = env.reset()
            # print(obs_list[0])
            # plt.imshow(obs_shape_list[0])
            rAll = 0
            for j in range(min_step):
                # 输入的是一个四维的数组，把真实的图像输入进去，返回一个模型
                allQ = qnetwork(obs_list[0]).numpy()
                #print("output:",allQ[0])
                # print("allq:", allQ)
                a = np.argmax(allQ, 1)
                # print("a:", a)
                # 使用的是onehot代码，把他转化为类似这样现状输出;输出的是一个二维shape(1,5) 的数组，类似[[1,2,3,4,5]]
                d_action = np.eye(d_action_dim, dtype=np.int32)[a]
                c_action = None
                # _obs_list 表示下一时刻的状态；env.step() 输入什么类型，看它的返回值
                _obs_list, reward, done, max_step = env.step(d_action, c_action)
                #experience pool=np.numpy(_obs_list, reward, done, max_step)
                # experience_pool1=[]
                # frame_count=0
                # experience_pool1.append((_obs_list,d_action,reward,done))
                # if frame_count % 4 == 0 and len(state_history) - 1 > batch_size:
                # # Select minibatch frames
                #     i_list = np.random.choice(range(len(state_history) - 1), size=batch_size)
                #     state_sample = np.array([state_history[i][0] for i in i_list])
                #     next_state_sample = np.array([state_history[i+1][0] for i in i_list])
                #     reward_sample = np.array([state_history[i][2] for i in i_list])
                #     done_sample = np.array([float(state_history[i][3]) for i in i_list])
                grt_action_step = np.argmax(d_action, 1)   # 取组最大值的下标
                # _obs_list[0],通过执行某个动作，看到了当前所处的状态，输出五个动作
                Q1 = qnetwork(_obs_list[0]).numpy()
                maxQ1 = np.max(Q1)                        # 选取最大的动作
                targetQ = allQ
                targetQ[0, grt_action_step[0]] = reward + lambd * maxQ1  # 更新目标网络，替换某个动作，相当于更新Q 表
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(obs_list[0])
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                # train_weights 相当于于一个θ，
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                rAll += reward
                ave_reward=rAll/min_step
                obs_list = _obs_list
                if done == True:
                    # reduce e, GLIE: Greey in the limit with infinite Exploration
                    e = 1. / ((i / 50) + 10)
                    break
            ave_reward = rAll/min_step
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                i + 1, num_episodes, ave_reward[0], time.time() - t0))

            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1]*0.9+rAll*0.1)
            with summary_writer.as_default():
                # tensorboard  --logdir=logs123
                tf.summary.scalar('reward', ave_reward[0], step=i)
            plt.plot(all_episode_reward)
        save_ckpt(qnetwork)  # save model
            # plt.plot(all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

    if args.test:
        load_ckpt(qnetwork)  # load model
        for i in range(num_episodes):
            # Reset environment and get first new observation
            obs_list = env.reset()  # observation is state, is picture
            rAll = 0
            for j in range(99):  # step index, maximum step is 99
                # Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(obs_list[0]).numpy()
            # Get new state and reward from environment
                a = np.argmax(allQ, 1)
                d_action = np.eye(d_action_dim, dtype=np.int32)[a]
                c_action = None
                _obs_list, reward, done, max_step = env.step(d_action, c_action)
                rAll += reward
                obs_list = _obs_list
                ave_reward = rAll/min_step
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
            i + 1, num_episodes, ave_reward[0], time.time() - t0))
