import torch
import gym
from dqn_agent import Agent
if __name__ == '__main__':
    agent = Agent(state_size=8,action_size=4,seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    env = gym.make('LunarLander-v2')
    env.seed(0)
    for i in range(3):
        state = env.reset()
        result=0
        print(result)
        for j in range(2000):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            result += reward

            if done:
                print("game reward:", result)
                break

    env.close()