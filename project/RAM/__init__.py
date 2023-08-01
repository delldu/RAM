"""Image Recognize Anything Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 01:55:56 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import todos
from . import ram
from torchvision.transforms import Compose, ToTensor
from pathlib import Path
import pdb

CONFIG_PATH=(Path(__file__).resolve().parents[0])


def tag_list(tag_file_name=f"{CONFIG_PATH}/data/ram_tag_list.txt"):
    print(f"Loading tag from {tag_file_name} ...")

    with open(tag_file_name, 'r', encoding="utf-8") as f:
        tag_list = f.read().splitlines()
    tag_list = np.array(tag_list)
    return tag_list

def create_model():
    """
    Create model
    """

    model = ram.RAM()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    # model = ram.RAM()
    # device = todos.model.get_device()
    # model = model.to(device)
    # model.eval()
    model, device = create_model()

    print(f"Running model on {device} ...")
    # print(model)

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/RAM.torch"):
        model.save("output/RAM.torch")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    transform = Compose([
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    english_tag_list = tag_list(f"{CONFIG_PATH}/data/ram_tag_list.txt")
    chinese_tag_list = tag_list(f"{CONFIG_PATH}/data/ram_tag_list_chinese.txt")

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
            tag_index = model(input_tensor).cpu()

        results.append(f"File name: {filename}")
        english_tags = ' | '.join(english_tag_list[tag_index])
        chinese_tags = ' | '.join(chinese_tag_list[tag_index])
        results.append(f"Image Tags: {english_tags}")
        results.append(f"图像标签: {chinese_tags}")
        results.append("-" * 128)
    progress_bar.close()

    print("\n".join(results))

    todos.model.reset_device()
