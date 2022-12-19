# 进度条显示

from tqdm import tqdm
import time
for i in tqdm(range(10000)):
    time.sleep(0.01)
    # print("开始打印")