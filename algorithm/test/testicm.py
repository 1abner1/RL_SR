import torch
import torch.nn as nn
class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 1
        self.action_net = nn.Sequential(
            nn.Linear(6, 3),  # 输入当前状态和下一时刻状态，输出预测动作a
            # nn.ReLU(),
            nn.Linear(3,feature_output)
        )
        self.state_net = nn.Sequential(
            nn.Linear(4, 512),
            nn.Linear(512, output_size)   # 输入为当前状态和当前的动作，预测出下一时刻的状态为3维
        )

    def forward(self, state, next_state, action):
        # state, next_state, action = inputs
        state = state.to(torch.float32)
        next_state = next_state.to(torch.float32)
        co_state = state.append(next_state)
        predi_action = self.action_net(co_state)
        state_action = state.append(action)
        pred_action = self.state_net(state_action)
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