import numpy as np
import torch
import sys
sys.path.append(r'D:\Pytorch_RL _sort\algorithm\deep_rl')
from algorithm.deep_rl.image_to_conv import CNNNet
import torch.optim as optim
import torch.nn.functional as F

# from common.utils import *
# from common.buffers import *
# from common.networks import *
from agents.common.utils import *
from agents.common.buffers import *
from agents.common.networks import *
import torchvision.transforms as transforms
# from env_wrapper.ros_car1 import ros_message_pb2
# print("开始导入roscar_server")
# from env_wrapper.ros_car1 import ros_message_pb2
# # print("ros_server 导入成功")
# import server
# print("ros环境导入成功")
# # form env_wrapper.ros_car.ros_message_pb2_grpc import *
# # from env_wrapper.ros_car.server import RosMessageServicer
# # print("ros环境导入成功")
# sr1 = server.RosMessageServicer()
# # start_obs = sr1.GetAction(self,request: ros_message_pb2.ObservationsRequest, context)
# # print("sr133333333333333333333333333",sr1.Init())
# # obs = sr1.vis
# # print("obs333",obs)
# from ros_wrapper import
#
# start_communcation1 = sr1.start_communcation()
# print("开始通信1111111111111111111",)
# start_obs = sr1.get_obs()
# print("获得star_obs",star_obs)


class Agent(object):
   """
   An implementation of the Proximal Policy Optimization (PPO) (by clipping) agent, 
   with early stopping based on approximate KL.
   """

   def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_dim,
                act_limit,
                steps=0,
                gamma=0.99,
                lam=0.97,
                hidden_sizes=(64,64),
                sample_size=2048,
                train_policy_iters=80,
                train_vf_iters=80,
                clip_param=0.2,
                target_kl=0.01,
                policy_lr=3e-4,
                vf_lr=1e-3,
                eval_mode=False,
                policy_losses=list(),
                vf_losses=list(),
                kls=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_limit = act_limit
      self.steps = steps 
      self.gamma = gamma
      self.lam = lam
      self.hidden_sizes = hidden_sizes
      self.sample_size = sample_size
      self.train_policy_iters = train_policy_iters
      self.train_vf_iters = train_vf_iters
      self.clip_param = clip_param
      self.target_kl = target_kl
      self.policy_lr = policy_lr
      self.vf_lr = vf_lr
      self.eval_mode = eval_mode
      self.policy_losses = policy_losses
      self.vf_losses = vf_losses
      self.kls = kls
      self.logger = logger

      # Main network
      self.policy = GaussianPolicy(self.obs_dim, self.act_dim, self.act_limit).to(self.device)
      self.vf = MLP(self.obs_dim, 1, activation=torch.tanh).to(self.device)
      
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)
      
      # Experience buffer
      self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.device, self.gamma, self.lam)

   def compute_vf_loss(self, obs, ret, v_old):
      # Prediction V(s)
      v = self.vf(obs).squeeze(1)

      # Value loss
      clip_v = v_old + torch.clamp(v-v_old, -self.clip_param, self.clip_param)
      vf_loss = torch.max(F.mse_loss(v, ret), F.mse_loss(clip_v, ret)).mean()
      return vf_loss

   def compute_policy_loss(self, obs, act, adv, log_pi_old):
      # Prediction logπ(s)
      _, _, _, log_pi = self.policy(obs, act, use_pi=False)
      
      # Policy loss
      ratio = torch.exp(log_pi - log_pi_old)
      clip_adv = torch.clamp(ratio, 1.-self.clip_param, 1.+self.clip_param)*adv
      policy_loss = -torch.min(ratio*adv, clip_adv).mean()

      # A sample estimate for KL-divergence, easy to compute
      approx_kl = (log_pi_old - log_pi).mean()
      return policy_loss, approx_kl

   def train_model(self):
      batch = self.buffer.get()
      obs = batch['obs']
      act = batch['act']
      ret = batch['ret']
      adv = batch['adv']
      
      # Prediction logπ_old(s), V_old(s)
      _, _, _, log_pi_old = self.policy(obs, act, use_pi=False)
      log_pi_old = log_pi_old.detach()
      v_old = self.vf(obs).squeeze(1)
      v_old = v_old.detach()

      # Train policy with multiple steps of gradient descent
      for i in range(self.train_policy_iters):
         policy_loss, kl = self.compute_policy_loss(obs, act, adv, log_pi_old)
         
         # Early stopping at step i due to reaching max kl
         if kl > 1.5 * self.target_kl:
            break
         
         # Update policy network parameter
         self.policy_optimizer.zero_grad()
         policy_loss.backward()
         self.policy_optimizer.step()
      
      # Train value with multiple steps of gradient descent
      for i in range(self.train_vf_iters):
         vf_loss = self.compute_vf_loss(obs, ret, v_old)

         # Update value network parameter
         self.vf_optimizer.zero_grad()
         vf_loss.backward()
         self.vf_optimizer.step()

      # Save losses
      self.policy_losses.append(policy_loss.item())
      self.vf_losses.append(vf_loss.item())
      self.kls.append(kl.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      # unity 获取的观察数据
      obs = self.env.reset()
      # print("obs[0][0]:", obs[0][0])
      obs_arry = np.array(obs[0][0])
      obs_tensor = torch.from_numpy(obs_arry)
      obs_tensor_input = obs_tensor.unsqueeze(dim=0)
      changge_obs = obs_tensor_input.view([1,3,84,84])
      # print("obs_tensor:",obs_tensor)
      # print("obs_tensor_input.shape ：", obs_tensor_input.shape)
      # print("changge_obs.shape ：", changge_obs.shape)
      # print("obs_arry 矩阵格式:", obs_arry)
      #ros 车交互获得图像信息
      # ros_env = RosMessageServicer(ros_message_pb2_grpc.RosMessageServicer)
      # ros_obs = ros_env.get_obs()
      # print("ros_obs:",ros_obs)

      #ros 获取的观察数据
      # import ros_wrapper
      # # 获取实际ros 小车的环境信息
      # ros_env = ros_wrapper.ros_wrapper()
      #
      # #输出的动作值
      # ros_obs = ros_env.vis
      # print("输出的观察值3333333333",ros_obs)
      # ros_obs_tensor = torch.from_numpy(ros_obs)
      # # print("输出tensor1111111111",ros_obs_tensor)
      # # print("输出tensor1111111111shape", ros_obs_tensor.shape)
      # #归一化
      # transformer = transforms.Compose([transforms.Resize((84, 84)),  # resize 调整图片大小
      #                                   # transforms.RandomHorizontalFlip(), # 水平反转
      #                                   transforms.ToTensor(),  # 0-255 to 0-1 归一化处理
      #                                   # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  #归一化
      #                                   ])
      # # 转化为图像格式
      # from PIL import Image
      # ros_obs1 = np.uint8(ros_obs)  # 这个没有使用，把代码转化为unint8的格式
      # # 非常重要11111111111111111111111 开启下面几行代码在小车上推断
      # ros_PIL_image = Image.fromarray(ros_obs)
      # # # 显示2s 图像
      # # from matplotlib import pyplot as plt
      # # # plt.switch_backend("agg")
      # # plt.imshow(ros_PIL_image)
      # # plt.show(block=False)
      # #
      # # plt.pause(0.05)
      # # plt.close()
      # # # 显示2s 图像
      # ros_guiyi = transformer(ros_PIL_image)
      # # print("归一化之后1111111111111",ros_guiyi)
      # ros_guiyi1 = ros_guiyi.view([1, 3, 84, 84])
      # # print("归一化之后shape", ros_guiyi1.shape)
      # # print("归一化之后的ros图像信息",ros_guiyi1)

      done = False
      net = CNNNet()
      # print("卷积神经网络初始化完成33333333333333")
      OUTPUT_obs = net.forward(changge_obs)  #changge_obs就是获得图像数据；ros_guiyi ros小车图像；
      out_obs_array = OUTPUT_obs[0]
      out_obs_array = out_obs_array.detach().numpy()
      input_obs= torch.from_numpy(out_obs_array)
      # print("输出卷积之后的类别",OUTPUT_obs)
      # print("OUTPUT_obs[0]", OUTPUT_obs[0])
      # print("input_obs.shape", input_obs.shape)
      # print("input_obs", input_obs)

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         if self.args.render:
            self.env.render()

         if self.eval_mode:
            action, _, _, _ = self.policy(torch.Tensor(obs[1][0]).to(self.device))
            action = action.detach().cpu().numpy()
            action = np.expand_dims(action, 0)
            next_obs, reward, done, _ = self.env.step(action,None)
         else:
            self.steps += 1
            # input_obs1 = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
            # input_obs1 = np.array(input_obs1)
            # input_obs1 = torch.FloatTensor(input_obs1)
            # print("input_obs1类型输出为:",input_obs1)
            # Collect experience (s, a, r, s') using some policy
            _, _, action, _ = self.policy(input_obs.to(self.device))
            # _, _, action, _ = self.policy(input_obs1.to(self.device))
            # print("策略输出成功222222222222")
            action = action.detach().cpu().numpy()
            action = np.expand_dims(action, 0)
            # print("action动作",action)
            # print("action动作shape", action.shape)
            # 开始传入到车子动作
            ros_action1_index = np.argmax(action, axis=1)
            # print("ros_action1",ros_action1_index)
            # print("ros_action1index[0]", ros_action1_index[0])
            # t123= ros_env.recv_action([ros_action1_index[0]])
            # print("t123",t123)
            # 把动作传个小车
            # ros_env.servicer.action =[ros_action1_index[0]]
            # # out_rosaction = ros_env.recv_action(ros_action1_index[0])
            # # print("out_rosaction333333",out_rosaction)
            # print("ros_env.servicer.action", ros_env.servicer.action)
            # ros_action = ros_action1_index[0]
            # 进入ros的实际环境
            # next_obs, reward, done, _ = ros_env.step(ros_action)
            # next_obs = input_obs.to(self.device)

            # 这一步后期需要修改为推断模式的话，就不需要和unity 交互了。
            next_obs, reward, done, _ = self.env.step(action,None) #就是把动作传给unity
            # print("reward1111111111111111111",reward)
            # Add experience to buffer,这里的观察值应该为下一步观察值的
            v = self.vf(input_obs.to(self.device))
            self.buffer.add(input_obs, action, reward, done, v)
            # print("运行了194步")
            
            # Start training when the number of experience is equal to sample size
            if self.steps == self.sample_size:
               self.buffer.finish_path()
               self.train_model()
               self.steps = 0

         total_reward += reward
         step_number += 1
         # obs = next_obs
         obs = input_obs
      
      # Save logs
      self.logger['LossPi'] = round(np.mean(self.policy_losses), 5)
      self.logger['LossV'] = round(np.mean(self.vf_losses), 5)
      self.logger['KL'] = round(np.mean(self.kls), 5)
      return step_number, total_reward
