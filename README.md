<img src="./algorithm/s2rlog/log.png" align="middle" width="2000"/>

## **Goal: Zero migration of the decision model in the virtual scene to the real scene guarantees good adaptivity and stability.**

# Environment
1)   [TORCS](https://github.com/ugo-nama-kun/gym_torcs)
2)   [UNTIY](https://github.com/Unity-Technologies/ml-agents)
3)   [GYM](https://github.com/openai/gym)


# Algorithm
1) AMDDPG
2) AMRL

# Requirement
1) python=3.9
2) mlagents==0.29.0
3) torch 
4) gym 
5) numpy==1.20.3
6) torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# How to runing
```python
1) python  amddg_run.py 
2) python  amrl_run.py

```
# Main paper


# Reference

* [CleanRL](https://github.com/vwxyzjn/cleanrl) is a learning library based on the Gym API. It is designed to cater to newer people in the field and provides very good reference implementations.
* [Tianshou](https://github.com/thu-ml/tianshou) is a learning library that's geared towards very experienced users and is design to allow for ease in complex algorithm modifications.
* [RLlib](https://docs.ray.io/en/latest/rllib/index.html) is a learning library that allows for distributed training and inferencing and supports an extraordinarily large number of features throughout the reinforcement learning space.


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


