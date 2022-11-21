<img src="./algorithm/s2rlog/log.png" align="middle" width="2000"/>

## **Goal: Zero migration of the decision model in the virtual scene to the real scene guarantees good adaptivity and stability.**

# Environment
1)   [TORCS](https://github.com/ugo-nama-kun/gym_torcs)
2)   [UNTIY](https://github.com/Unity-Technologies/ml-agents)
3)   [GYM](https://github.com/openai/gym)


# Algorithm
1) AMDDPG
2) AMRL
3) PPO
4) TRPO
5) SAC
6) MAML
7) DDPG
8) RL^2
9) EPG
10) DQN
11) DDQN



# Requirement
1) python=3.9
2) mlagents==0.29.0
3) torch 
4) gym 
5) numpy==1.20.3
6) torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Function
1) reinforcement learning 
2) target detection
3) Semantic segmentation



# How to runing
```python
1) python  amddg_run.py 
2) python  amrl_run.py

```
# Main paper
* [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
* [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
* [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
* [Intrinsic Curiosity Module (ICM)](https://arxiv.org/pdf/1705.05363.pdf)


# Reference
* [CleanRL](https://github.com/vwxyzjn/cleanrl) is a learning library based on the Gym API. It is designed to cater to newer people in the field and provides very good reference implementations.
* [Tianshou](https://github.com/thu-ml/tianshou) is a learning library that's geared towards very experienced users and is design to allow for ease in complex algorithm modifications.
* [RLlib](https://docs.ray.io/en/latest/rllib/index.html) is a learning library that allows for distributed training and inferencing and supports an extraordinarily large number of features throughout the reinforcement learning space.
* [Ray/Lilib](https://github.com/ray-project/ray/tree/master/rllib/) Ray is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries (Ray AIR) for simplifying ML compute.

# Citation

```
@article{xiao2022feature,
  title={Feature semantic space-based sim2real decision model},
  author={Xiao, Wenwen and Luo, Xiangfeng and Xie, Shaorong},
  journal={Applied Intelligence},
  pages={1--17},
  year={2022},
  publisher={Springer}
}
```

## License
[Apache License 2.0](LICENSE.md)


