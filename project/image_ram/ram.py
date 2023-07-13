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

from .bert import BertConfig, BertModel
from .swin_transformer import SwinTransformer

from transformers import BertTokenizer
from pathlib import Path

CONFIG_PATH=(Path(__file__).resolve().parents[0])


def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


class RAM(nn.Module):
    def __init__(self,
                 image_size=384,
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list=f'{CONFIG_PATH}/data/ram_tag_list.txt',
                 tag_list_chinese=f'{CONFIG_PATH}/data/ram_tag_list_chinese.txt'):
        r""" The Recognize Anything Model (RAM) inference module.
        RAM is a strong image tagging model, which can recognize any common category with high accuracy.
        Described in the paper " Recognize Anything: A Strong Image Tagging Model" https://recognize-anything.github.io/
        
        Args:
            image_size (int): input image size
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
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

        # create tokenzier
        self.tokenizer = init_tokenizer()

        self.delete_tag_index = delete_tag_index

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)
        self.tag_list_chinese = self.load_tag_list(tag_list_chinese)

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file(f'{CONFIG_PATH}/config/q2l_config.json')
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config)
        # self.tagging_head.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = nn.Parameter(torch.zeros(self.num_class, q2l_config.encoder_width))
        self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size) # q2l_config.hidden_size -- 768

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        self.image_proj = nn.Linear(vision_width, 512)

        # adjust thresholds for some tags
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        ram_class_threshold_path = f'{CONFIG_PATH}/data/ram_tag_list_threshold.txt'
        with open(ram_class_threshold_path, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key,value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

        self.load_weights()

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        # del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def forward(self, image):
        # return image

        label_embed = F.relu(self.wordvec_proj(self.label_embed))


        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]


        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )

        logits = self.fc(tagging_embed).squeeze(-1)


        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        tag = targets.cpu().numpy()
        tag[:,self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(' | '.join(token_chinese))


        return tag_output, tag_output_chinese


    def load_weights(self, model_path="models/image_ram.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
