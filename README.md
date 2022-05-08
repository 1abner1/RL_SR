# RL_SR
# run AMRL 
# install python=3.9
#pip install mlagents==0.29.0
#pip install torch gym numpy==1.20.3
4.使用cuda 10.2  pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

office train:
mlagents-learn "D:\RL_SR\ml-agents-release_1703\config\ppo\carconavoid.yaml"  --env="D:\RL_SR\envs\test\car_seg_avoid.exe" --width=1000   --height=1000  --num-envs=1 --torch-device "cuda"  --time-scale=20

