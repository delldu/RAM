'''
 * The Recognize Anything Model (RAM)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random
import time

import torch

from PIL import Image
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform


parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/1641173_2291260800.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()
    # torch.save(model.state_dict(), "/tmp/image_ram.pth") # 823M    
    model = model.to(device)

    print(model)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    start_time = time.time()
    res = inference(image, model)
    print("Spend time: ", time.time() - start_time)

    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])
