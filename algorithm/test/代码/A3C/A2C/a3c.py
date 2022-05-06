import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import os
import torch.multiprocessing as mp
from models import TwoHeadNetwork
import matplotlib.pyplot as plt
from utils import v_wrap,set_init,push_and_pull,record
UPDATE_GLOBAL_ITER = 5
os.environ["OMP_NUM_THREADS"] = "1"
GAMMA = 0.99
MAX_EP = 30000
class SharedAdam(optim.Adam):
    def __init__(self,params,lr=1e-4,betas=(0.9,0.99),eps=1e-8,weight_decay=0):
        super(SharedAdam, self).__init__(params,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)


                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

class Net(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.policy1 = nn.Linear(s_dim,256)
        self.policy2 = nn.Linear(256,a_dim)

        self.value1 = nn.Linear(s_dim,256)
        self.value2 = nn.Linear(256,1)
        set_init([self.policy1,self.policy2,self.value1,self.value2])
        self.distribution = torch.distributions.Categorical
    def forward(self,x):
        logits = F.relu(self.policy1(x))
        logits = self.policy2(logits)

        value = F.relu(self.value1(x))
        value = self.value2(value)

        return logits, value

    def choose_action(self,s):
        self.eval()
        logits,_ = self.forward(s)
        prob = F.softmax(logits,dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self,s,a,v_t):
        self.train()
        logits,values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits,dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a)*td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss+a_loss).mean()
        return total_loss
class Worker(mp.Process):
    def __init__(self,gnet,opt,global_ep,global_ep_r,res_queue,name):
        super(Worker, self).__init__()
        self.name = 'w%02i'%name
        self.g_ep,self.g_ep_r,self.res_queue = global_ep,global_ep_r,res_queue
        self.gnet,self.opt = gnet,opt
        self.lnet = Net(N_S,N_A)
        self.env = gym.make('CartPole-v0').unwrapped
    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s,buffer_a,buffer_r = [],[],[]
            ep_r = 0.
            while True:
                '''if self.name == 'w00':
                    self.env.render()'''
                a=self.lnet.choose_action(v_wrap(s[None,:]))
                s_,r,done,_ = self.env.step(a)
                if done:r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                if total_step%UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt,self.lnet,self.gnet,done,s_,buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s,buffer_a,buffer_r = [],[],[]
                    if done:
                        record(
                            self.g_ep,self.g_ep_r,ep_r,self.res_queue,self.name
                        )
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

if __name__ == '__main__':
    gnet = Net(N_S,N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(),lr = 1e-4,betas=(0.92,0.999))
    global_ep,global_ep_r,res_queue = mp.Value('i',0),mp.Value('d',0.),mp.Queue()

    workers = [Worker(gnet,opt,global_ep,global_ep_r,res_queue,i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()