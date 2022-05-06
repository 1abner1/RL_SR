import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ICM():
    def __init__(self, state_converter: Converter, action_converter: Converter, model_factory: ICMModelFactory,
                 policy_weight: float, reward_scale: float, weight: float, intrinsic_reward_integration: float,
                 reporter: Reporter):
        super().__init__(state_converter, action_converter)
        self.model: ICMModel = model_factory.create(state_converter, action_converter)   #返回状态的state_converter 和 动作的converter
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        return self.model.parameters()

    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:, :-1], states[:, 1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states, self.model.recurrent),
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:]))
        next_states_latent, next_states_hat, _ = self.model(states, next_states, actions)
        intrinsic_reward = self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy().reshape(n, t)
        self.reporter.scalar('icm/reward',
                             intrinsic_reward.mean().item() if self.reporter.will_report('icm/reward') else 0)
        return (1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_reward

    def loss(self, policy_loss: Tensor, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        next_states_latent, next_states_hat, actions_hat = self.model(states, next_states, actions)
        forward_loss = 0.5 * (next_states_hat - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()   # 最重要的就是这两个loss,状态loss
        inverse_loss = self.action_converter.distance(actions_hat, actions)                                  # 动作loss,这两个loss.
        curiosity_loss = self.weight * forward_loss + (1 - self.weight) * inverse_loss                       # 将两个loss 和在一起
        self.reporter.scalar('icm/loss', curiosity_loss.item())
        return self.policy_weight * policy_loss + curiosity_loss

    def to(self, device: torch.device, dtype: torch.dtype):
        super().to(device, dtype)
        self.model.to(device, dtype)

    @staticmethod
    def factory(model_factory: ICMModelFactory, policy_weight: float, reward_scale: float,
                weight: float, intrinsic_reward_integration: float, reporter: Reporter = NoReporter()) -> 'ICMFactory':
        return ICMFactory(model_factory, policy_weight, reward_scale, weight, intrinsic_reward_integration, reporter)