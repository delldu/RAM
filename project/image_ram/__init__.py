"""Image/Video Segment Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image

import torch
import todos
from . import ram
from torchvision.transforms import Normalize, Compose, Resize, ToTensor

def get_transform(image_size=384):
    return Compose([
        lambda image: image.convert("RGB"),
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


import pdb

def get_tvm_model():
    """
    TVM model base on torch.jit.trace
    """

    model = ram.RAM()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


def get_ram_model():
    """Create model."""

    model = ram.RAM()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")
    # print(model)
    
    model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_ram.torch"):
    #     model.save("output/image_ram.torch")

    return model, device


def image_ram_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_ram_model()
    transform = get_transform()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    results = []
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            eng_tags, chinese_tags = model(input_tensor)

        results.append(f"File name: {filename}")
        results.append(f"Image Tags: {eng_tags}")
        results.append(f"图像标签: {chinese_tags}")
        results.append("-" * 128)
    progress_bar.close()

    print("\n".join(results))

    todos.model.reset_device()
