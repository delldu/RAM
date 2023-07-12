# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
import torch
import image_ram

from tqdm import tqdm

if __name__ == "__main__":
    model, device = image_ram.get_ram_model()

    N = 100
    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W  # model.max_h, model.max_w

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        x = torch.randn(B, C, H, W)
        # print("x: ", x.size())

        start_time = time.time()
        with torch.no_grad():
            y, _ = model(x.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")
