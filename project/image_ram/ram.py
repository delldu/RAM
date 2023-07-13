'''
 * The Recognize Anything Model (RAM)
 * Written by Xinyu Huang
'''
import os
import json
import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.configuration_bert import BertConfig
from pathlib import Path

from .bert import BertModel
from .swin_transformer import SwinTransformer

CONFIG_PATH=(Path(__file__).resolve().parents[0])

def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)

class RAM(nn.Module):
    def __init__(self,
                 image_size=384,
                 threshold=0.68,
                 tag_list=f'{CONFIG_PATH}/data/ram_tag_list.txt'):
        r""" The Recognize Anything Model (RAM) inference module.
        RAM is a strong image tagging model, which can recognize any common category with high accuracy.
        Described in the paper " Recognize Anything: A Strong Image Tagging Model" https://recognize-anything.github.io/
        
        Args:
            image_size (int): input image size
            threshold (int): tagging threshold
        """
        super().__init__()
        vision_config_path = f'{CONFIG_PATH}/config/config_swinL_384.json'
        vision_config = read_json(vision_config_path)
        assert image_size == vision_config['image_res']
        # assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        self.visual_encoder = SwinTransformer(
            img_size=vision_config['image_res'],
            patch_size=4,
            in_chans=3,
            embed_dim=vision_config['embed_dim'],
            depths=vision_config['depths'],
            num_heads=vision_config['num_heads'],
            window_size=vision_config['window_size'],
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1)

        # create image-tag recognition decoder
        self.num_class = len(self.load_tag_list(tag_list)) # load tag list
        q2l_config = BertConfig.from_json_file(f'{CONFIG_PATH}/config/q2l_config.json')
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config)
        self.label_embed = nn.Parameter(torch.zeros(self.num_class, q2l_config.encoder_width))
        self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size) # q2l_config.hidden_size -- 768

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        self.image_proj = nn.Linear(vision_width, 512)

        # adjust thresholds for some tags
        class_threshold = torch.ones(self.num_class) * threshold
        ram_class_threshold_path = f'{CONFIG_PATH}/data/ram_tag_list_threshold.txt'
        with open(ram_class_threshold_path, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key,value in enumerate(ram_class_threshold):
            class_threshold[key] = value
        self.class_threshold = class_threshold

        self.load_weights()


    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def forward(self, image):
        label_embed = F.relu(self.wordvec_proj(self.label_embed))
        image_embeds = self.image_proj(self.visual_encoder(image))
        B, C, HW = image_embeds.size()
        image_atts = torch.ones((B, C), dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        label_embed = label_embed.unsqueeze(0).repeat(B, 1, 1)

        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        ) # [1, 4585, 768]

        logits = self.fc(tagging_embed).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        _, index = torch.where(targets == 1)
        return index


    def load_weights(self, model_path="models/image_ram.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"model weight file '{checkpoint}'' not exist !!!")
