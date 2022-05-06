import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import os
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

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

def v_wrap(np_array, dtype=np.float32):
     if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
     return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
    buffer_v_target = []
    for r in br[::-1]:
        v_s = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()
    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None])
    )
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r:%.0f" % global_ep_r.value,
    )

UPDATE_GLOBAL_ITER = 5
os.environ["OMP_NUM_THREADS"] = "1"
GAMMA = 0.99
MAX_EP = 30000
env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

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

                self.env.render()
                a= self.lnet.choose_action(v_wrap(s[None,:]))
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

def train():
    pass

def test():
    pass

if __name__ == '__main__':

    gnet = Net(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
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
