import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime
import gym
from torch.utils.tensorboard import SummaryWriter
from models import PolicyNetwork,ValueNetwork
from torch.distributions import Categorical

class A2CAgent():
    def __init__(self,env,gamma,lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.value_network = ValueNetwork(self.obs_dim,1).to(self.device)
        self.policy_network = PolicyNetwork(self.obs_dim,self.action_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(),lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(),lr=self.lr)
    def get_action(self,state):
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy_network.forward(state)
        dist = F.softmax(logits,dim=0)
        probs = Categorical(dist)
        return probs.sample().cpu().detach().item()

    def compute_loss(self,trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)

        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device)*rewards[j:])for j in range(rewards.size(0))]
        value_targets = rewards.view(-1,1) + torch.FloatTensor(discounted_rewards).view(-1,1).to(self.device)
        values = self.value_network.forward(states)
        value_loss = F.mse_loss(values,value_targets.detach())

        logits = self.policy_network.forward(states)
        dists = F.softmax(logits,dim=1)
        probs = Categorical(dists)

        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean()*torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1,1)*advantage.detach()
        policy_loss = policy_loss.mean() - 0.001*entropy
        return value_loss,policy_loss
    def update(self,trajectory,writer,n_iter):
        value_loss,policy_loss = self.compute_loss(trajectory)
        writer.add_scalar('value_loss', float(value_loss), n_iter)
        writer.add_scalar('policy_loss', float(policy_loss), n_iter)
        self.value_optimizer.zero_grad()
        value_loss.backward()

        self.value_optimizer.step()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    MAX_EPISODE = 5000
    MAX_STEPS = 5000
    lr = 1e-4
    gamma = 0.99
    agent = A2CAgent(env,gamma,lr)
    now = datetime.datetime.now()
    date_time1 = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    writer = SummaryWriter(log_dir="runs/A2Cde"+"CartPole-v0"+'_'+date_time1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iter = 0
    episode_reward_over100 = 0
    start_time = datetime.datetime.now()
    for episode in range(MAX_EPISODE):
        iter += 1
        state = env.reset()
        trajectory = []
        episode_reward = 0

        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward
            if done:
                break
            state = next_state
        writer.add_scalar('reward', float(episode_reward), episode)
        episode_reward_over100 += episode_reward
        if (episode % 100 == 0):
            episode_reward_over100 = episode_reward_over100 / 100.0
            writer.add_scalar('averageR', float(episode_reward_over100), episode / 100)
            writer.add_scalar("time_reward", float(episode_reward_over100),
                              (datetime.datetime.now() - start_time).total_seconds())
            print("average reward is {}".format(episode_reward_over100))
            if (episode_reward_over100 >= 195.0):
                print("The agent spent {} episodes finishing this task".format(episode))
                break
            episode_reward_over100 = 0
        if episode % 10 == 0:
            print("Episode" + str(episode) + ":" + str(episode_reward))

        agent.update(trajectory, writer, iter)
    writer.close()