# from functools import reduce
#
# def mul(a, b):
#     "Same as a * b."
#     return a * b
# def reduce_1():
#     x = reduce(mul,(125,),1)
#     return x
# x=reduce_1()
# print("x的输出",x)
import gym
# env = gym.make("AntVel-v2",low=0.0,high=3.0,normalization_scale=10.0,max_episode_steps=100)
env = gym.make("Ant-v2")
for _ in range(1000):
   env.render()