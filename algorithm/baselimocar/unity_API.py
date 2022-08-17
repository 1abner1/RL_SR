import logging
import itertools

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig, EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

logger = logging.getLogger('UnityWrapper')
logger.setLevel(level=logging.INFO)


class UnityWrapper:
    def __init__(self,
                 train_mode=False,  # False的话渲染比较流畅，可以看训练效果
                 file_name=None, #r"D:\xwwppounityrollerimage\unitycar\car_nvigation.exe",
                 base_port=5004,
                 no_graphics=False,
                 seed=None,
                 scene=None,
                 n_agents=1):

        self.scene = scene

        seed = seed if seed is not None else np.random.randint(0, 65536)

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self._env = UnityEnvironment(file_name=file_name,
                                     base_port=base_port,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed)

        if train_mode:
            self.engine_configuration_channel.set_configuration_parameters(width=480,
                                                                           height=480,
                                                                           quality_level=2,
                                                                          time_scale=20)
            print("设置宽度运行了11111111111111111111")
        else:
            self.engine_configuration_channel.set_configuration_parameters(width=2,
                                                                           height=2,
                                                                           quality_level=5,
                                                                           time_scale=0.01,
                                                                           target_frame_rate=60)

        self._env.reset()
        self.bahavior_name = list(self._env.behavior_specs)[0]

    def init(self):
        behavior_spec = self._env.behavior_specs[self.bahavior_name]
        obs_shapes = [o.shape for o in behavior_spec.observation_specs]
        logger.info(f'Observation shapes: {obs_shapes}')

        self._empty_action = behavior_spec.action_spec.empty_action

        discrete_action_size = 0
        if behavior_spec.action_spec.discrete_size > 0:
            discrete_action_size = 1
            action_product_list = []
            for action, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                discrete_action_size *= branch_size
                action_product_list.append(range(branch_size))
                logger.info(f"Discrete action branch {action} has {branch_size} different actions")

            self.action_product = np.array(list(itertools.product(*action_product_list)))

        continuous_action_size = behavior_spec.action_spec.continuous_size

        logger.info(f'Continuous action size: {continuous_action_size}')

        self.d_action_dim = discrete_action_size
        self.c_action_dim = continuous_action_size

        for o in behavior_spec.observation_specs:
            if len(o) >= 3:
                self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                break

        return behavior_spec.observation_specs, discrete_action_size, continuous_action_size

    def reset(self, reset_config=None):
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset() #自己调用自己
        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        return [obs.astype(np.float32) for obs in decision_steps.obs]

    # def step(self,next_state,reward,done,_):
    #     ne
    def step(self, d_action, c_action):
        if self.d_action_dim:
            d_action = np.argmax(d_action, axis=1)
            d_action = self.action_product[d_action]

        self._env.set_actions(self.bahavior_name,
                              ActionTuple(continuous=c_action, discrete=d_action))
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)

        tmp_terminal_steps = terminal_steps

        while len(decision_steps) == 0:
            self._env.set_actions(self.bahavior_name, self._empty_action(0))
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.bahavior_name)
            tmp_terminal_steps.agent_id = np.concatenate([tmp_terminal_steps.agent_id,
                                                          terminal_steps.agent_id])
            tmp_terminal_steps.reward = np.concatenate([tmp_terminal_steps.reward,
                                                        terminal_steps.reward])
            tmp_terminal_steps.interrupted = np.concatenate([tmp_terminal_steps.interrupted,
                                                             terminal_steps.interrupted])

        reward = decision_steps.reward
        reward[tmp_terminal_steps.agent_id] = tmp_terminal_steps.reward

        done = np.full([len(decision_steps), ], False, dtype=bool)
        done[tmp_terminal_steps.agent_id] = True

        max_step = np.full([len(decision_steps), ], False, dtype=bool)
        max_step[tmp_terminal_steps.agent_id] = tmp_terminal_steps.interrupted

        return ([obs.astype(np.float32) for obs in decision_steps.obs],
                decision_steps.reward.astype(np.float32),
                done,
                max_step)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # print("2222222222222222222222222")
    env = UnityWrapper(train_mode=True, base_port=5004,file_name=r"D:\RL_SR\envs\limocar\AURP.exe")  #没有build出来之后的环境只能使用5004端口
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    # print("obs_shape_list", obs_shape_list)
    # print("type:", type(obs_shape_list[0]))
    print("dim:", len(obs_shape_list[0]))
    print("d_action_dim:", d_action_dim)
    print("c_action_dim:", c_action_dim)

    for episode in range(100):
        obs_list = env.reset()
        # print("obs_list图像数据111111111111111111111", obs_list[0][0])
        # print("obs_list雷达数据222222222222222222", obs_list[1][0])
        # print("obs_list位置速度数据3333333333333333", obs_list[2][0])
        n_agents = obs_list[0].shape[0]
        print("n_agents",n_agents)
        for step in range(100):
            d_action, c_action = None, None
            if d_action_dim:
                d_action = np.random.randint(0, d_action_dim, size=n_agents)
                print("d_action:", d_action)  # d_action: [4]
                d_action = np.eye(d_action_dim, dtype=np.int32)[d_action]
            if c_action_dim:
                c_action = np.random.randn(n_agents, c_action_dim)
                print("c_action:",c_action)
            print("d_action",d_action)
            obs_list, reward, done, max_step = env.step(d_action, c_action)
            print("执行完该步骤")

    env.close()
