import os
import sys
from einops import rearrange
import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision.models import resnet50, ResNet50_Weights, ResNet101_Weights, resnet101
import dinov2.utils.utils as dinov2_utils
from dinov2.models import vision_transformer as vits

import model.resnet as models
import model.vgg as vgg_models
from model.PPM import PPM

import argparse
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.args = args

        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.vgg = args.vgg
        self.dinov2 = True if args.layers == 'dinov2' else False
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.pretrained = True
        if self.dataset in ['pascal', 'FSS1000']:
            self.classes = 16
        elif self.dataset in ['coco', 'lvis', 'paco_part', 'pascal_part']:
            self.classes = 61
        else:
            raise ValueError('Unknown dataset: {}'.format(self.dataset))
        
        assert self.layers in [50, 101, 152, 'dinov2']
    
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif self.dinov2:
            print('INFO: Using DINOv2 vitl-14')

            dinov2_kwargs = dict(
                img_size=518,
                patch_size=14,
                init_values=1e-5,
                ffn_layer='mlp',
                block_chunks=0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
            )

            # Load DINOv2 ViT-L/14 distilled model
            dinov2: vits.DinoVisionTransformer = vits.__dict__[args.dinov2_size](**dinov2_kwargs)
            dinov2_utils.load_pretrained_weights(dinov2, args.dinov2_weights, "teacher")

            # Load linear classification head weights
            linear_head = nn.Linear(dinov2.embed_dim * 2, 1000)
            linear_head.load_state_dict(torch.load(args.dinov2_linear_head))

            # Split ViT into layers (ViT doesn't have layers like ResNet, but you can group blocks)
            self.layer0 = dinov2.patch_embed
            self.cls_token = dinov2.cls_token
            self.pos_embed = dinov2.pos_embed
            self.interpolate_pos_encoding = dinov2.interpolate_pos_encoding
            self.layer1 = nn.Sequential(*dinov2.blocks[:6])
            self.layer2 = nn.Sequential(*dinov2.blocks[6:12])
            self.layer3 = nn.Sequential(*dinov2.blocks[12:18])
            self.layer4 = nn.Sequential(*dinov2.blocks[18:24])
            self.fc = linear_head

        else:

            print('INFO: Using ResNet {}'.format(self.layers))
            if self.layers == 50:
                # resnet = models.resnet50(pretrained=self.pretrained)
                resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
                self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                self.avgpool, self.fc = resnet.avgpool, resnet.fc
            elif self.layers == 101:
                # resnet = models.resnet101(pretrained=self.pretrained)
                resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
                self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
                self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                self.avgpool, self.fc = resnet.avgpool, resnet.fc



        # Base Learner
        self.encoder = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)
        if self.vgg:
            fea_dim = 512
        elif self.dinov2:
            fea_dim = 1024
        else:
            fea_dim = 2048
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.classes, kernel_size=1))

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
            {'params': model.ppm.parameters()},
            {'params': model.cls.parameters()},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def forward(self, x):

        if self.args.layers in [50, 101]:
            x = self.encoder(x)
            x_cls = self.avgpool(x)
            x_cls = x_cls.flatten(1)
            x_cls = self.fc(x_cls)
            return x_cls
        
        elif self.args.layers == 'dinov2':
            B, C, H, W = x.shape
            x = self.layer0(x)  # (B, 1024, H/16, W/16)
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, H, W)
            x = self.layer1(x)  # (B, N, 1024)
            x = self.layer2(x)  # (B, N, 1024)
            x = self.layer3(x)  # (B, N, 1024)
            x = self.layer4(x)  # (B, N, 1024)

            cls_token = x[:, 0]
            patch_tokens = x[:, 1:]

            linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
            x_cls = self.fc(linear_input)  # (B, 1000)
            return x_cls

if __name__ == "__main__":

    
    class Args:
        zoom_factor = 8
        vgg = False
        data_set = 'paco_part'
        ignore_label = 255
        layers = 'dinov2'
        dinov2_size = 'vit_large'
        dinov2_weights = 'initmodel/dinov2/dinov2_vitl14_pretrain.pth'
        dinov2_linear_head = 'initmodel/dinov2/dinov2_vitl14_linear_head.pth'

    args = Args()
    model = OneModel(args).cuda()

    # 读取图片并转为tensor
    img_path = "/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets/VOC2012/JPEGImages/2007_000648.jpg"
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])
    input: torch.Tensor = transform(img).unsqueeze(0).cuda()  # shape: (1, 3, 518, 518)
    mask_path: Image = "/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets/VOC2012/SegmentationClass/2007_000648.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print('mask unique', np.unique(mask))   # [  0  38  89 128 147 220] 38: plane, 89: bus, 147: person
    mask[mask != 38] = 0
    mask[mask == 38] = 1
    mask = torch.from_numpy(mask).unsqueeze(0).cuda()
    mask = transforms.Resize((518, 518), interpolation=InterpolationMode.NEAREST)(mask)      # shape: (1, 518, 518)

    # 1=aeroplane, 6=bus, 15=person
    print('mask unique', mask.unique())

    # 输出结果
    input = input * mask
        
    # 保存图片
    save_img = transforms.ToPILImage()(input[0].cpu())
    save_img.save("input_saved.jpg")

    B, C, H, W = input.shape
    mask = rearrange(mask.unsqueeze(0), "b c h w -> b (h w) c")

    supp_feat_0 = model.layer0(input)
    supp_feat_0 = torch.cat((model.cls_token.expand(B, -1, -1), supp_feat_0), dim=1)
    supp_feat_0 = supp_feat_0 + model.interpolate_pos_encoding(supp_feat_0, H, W)

    supp_feat_1 = model.layer1(supp_feat_0)
    supp_feat_2 = model.layer2(supp_feat_1)
    supp_feat_3 = model.layer3(supp_feat_2)
    supp_feat_4 = model.layer4(supp_feat_3)  # (B, N, 1024)

    cls_token = supp_feat_4[:, 0]
    patch_tokens = supp_feat_4[:, 1:]

    linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
    supp_cls = model.fc(linear_input)  # (B, 1000)
    
    print("Output shape:", supp_cls.shape, supp_cls.argmax(dim=1))