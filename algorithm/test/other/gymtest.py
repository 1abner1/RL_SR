import gym
import time
import numpy as np
env = gym.make('Pendulum-v0')
action = [0]
observation = env.reset()  #状态
actions = np.linspace(-2, 2, 10)
for t in range(1000):  #
    # action[0] =  random.uniform(-2,2)   #力矩  -2到2
    action[0] = 2
    observation, reward, done, info = env.step(action)
    # print(action, reward, done)
    env.render()

env.close()