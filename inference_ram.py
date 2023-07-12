'''
 * The Recognize Anything Model (RAM)
 * Written by Xinyu Huang
'''
import argparse
import time
import torch
import pdb

from PIL import Image
from ram.models import ram
from ram import inference_ram # as inference
from ram import get_transform
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/*.jpg')
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

    # print(model)

    image_filenames = sorted(glob.glob(args.image))

    results = []
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        image = transform(Image.open(filename)).unsqueeze(0).to(device)
        start_time = time.time()
        res = inference_ram(image, model)
        results.append(f"Parsing {filename} Spend time: {time.time() - start_time:.4f} seconds")
        results.append(f"Image Tags: {res[0]}")
        results.append(f"图像标签: {res[1]}")
        results.append("-" * 64)
    progress_bar.close()

    print("\n".join(results))