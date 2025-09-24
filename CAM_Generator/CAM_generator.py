import math
import os, sys
from pathlib import Path
from time import time
from typing import Dict, List, Union
import cv2
from matplotlib import pyplot as plt
import numpy as np
from os.path import join as ospj
from os.path import dirname as ospd
from einops import rearrange, repeat

import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from torch import nn
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torchvision.models import resnet, ResNet50_Weights, ResNet101_Weights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from CAM_Generator.base.utils import PseudoMaskGeneration, generate_vis, t2n
from CAM_Generator.CIM.CIM import CIM

from CAM_Generator.attribution import CAMWrapper, GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, ScoreCAM, LayerCAM, AblationCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
from CAM_Generator.attribution.utils import normalize_saliency, visualize_single_saliency
from util.util import BNC2BCHW

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]

def BNC2BCHW(x: torch.Tensor):
    B, N, C = x.shape
    H = W = int(math.sqrt(N))
    assert N == H*W
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x

def Generator_CAM(
        args,
        cam_net: CAMWrapper,
        batch: Dict[str, torch.Tensor],
        threshold: Union[float, str],
        vis=False, 
        img_fore_cls=None,
        vis_path=None,
        ):
    '''
    vis: save images in log/vis/x/
    img_fore_cls: support image class, in torch.Tensor shape (b, 1)
    vis_path: the path for visualization

    return:
    s_cams: (b, h, w)
    q_cams: (b, h, w)
    s_masks:(b, 1, h, w)
    s_pseudo_masks:(b, 1, h, w)
    q_pseudo_masks:(b, 1, h, w)
    '''

    s_imgs: torch.Tensor = batch['support_imgs']      # [b,c,h,w]

    s_masks = batch['support_masks']    # [b,1,h,w]

    q_imgs = batch['query_img']         # [b,c,h,w]

    q_masks = batch['query_mask']       # [b,h,w]

    if img_fore_cls is None:    # else using existing img_fore_cls
        cim = CIM(later_mask=True)
        cim.eval()
        _, foreground_probability = cim(s_imgs, s_masks)
        img_fore_cls = torch.argmax(foreground_probability, dim=1)
    
    s_cams_layer3 = normalize_saliency(cam_net.get_mask(s_imgs, img_fore_cls, 'layer3'))
    s_cams = s_cams_layer3
    s_pseudo_masks, threshold = PseudoMaskGeneration(s_cams.squeeze(dim=1), threshold, original_img=s_imgs, s_mask=s_masks)
    s_cams = s_cams.clone().detach()
    s_pseudo_masks = torch.round(s_pseudo_masks.clone().detach()).bool()
    s_masks = s_masks.clone().detach()

    q_cams_layer3 = normalize_saliency(cam_net.get_mask(q_imgs, img_fore_cls, 'layer3'))
    q_cam = q_cams_layer3
    q_cam = q_cam.clone().detach()
    q_pseudo_masks, _ = PseudoMaskGeneration(q_cam.squeeze(dim=1), threshold=threshold, original_img=q_imgs)
    q_pseudo_masks = torch.round(q_pseudo_masks.clone().detach()).bool()

    if vis:     # only visualize for first support image
        for i in range(s_cams.size(0)):
            q_camfore = q_cam[i][0]     # [H,W]
            q_img = q_imgs[i]
            q_mask = q_masks[i]
            q_pseudo_mask = q_pseudo_masks[i]
            q_name = batch['q_name'][i]

            q_vis_image = torch.zeros_like(q_img, dtype=torch.uint8)
            for c in range(q_img.shape[0]):
                q_img_c = q_img[c]
                q_img_c = (q_img_c - q_img_c.min()) / (q_img_c.max() - q_img_c.min()) * 255
                q_vis_image[c] = q_img_c.to(torch.uint8)
            q_camfore_normalized = t2n(q_camfore)
            q_pseudo_mask = t2n(q_pseudo_mask)
            q_mask = t2n(q_mask.long())

            q_vis_image = np.int64(q_vis_image.detach().cpu())
            q_vis_image[q_vis_image > 255] = 255
            q_vis_image[q_vis_image < 0] = 0
            q_vis_image = np.uint8(q_vis_image.transpose(1,2,0))
            q_vis_path = ospj(vis_path, q_name, 'q')

            if not os.path.exists(ospd(q_vis_path)):
                os.makedirs(ospd(q_vis_path))

            plt.imsave(ospj(q_vis_path)+f"_camfore.png", generate_vis(q_camfore_normalized, q_vis_image, color='cam'))
            plt.imsave(ospj(q_vis_path)+f"_cammask.png", generate_vis(q_pseudo_mask, q_vis_image, rate=0.7, color='r'))    

            s_camfore = s_cams[i][0]    # [H,W]
            s_img = s_imgs[i]       # [c,h,w]
            s_mask = s_masks[i]     # [1,h,w]
            s_name = batch['s_name'][i]

            s_vis_image = torch.zeros_like(s_img, dtype=torch.uint8)
            for c in range(s_img.shape[0]):
                s_img_c = s_img[c]
                s_img_c = (s_img_c - s_img_c.min()) / (s_img_c.max() - s_img_c.min()) * 255
                s_vis_image[c] = s_img_c.to(torch.uint8)
            # save results
            s_vis_image = np.int64(s_vis_image.detach().cpu())
            s_vis_image[s_vis_image > 255] = 255
            s_vis_image[s_vis_image < 0] = 0
            s_vis_image = np.uint8(s_vis_image.transpose(1,2,0))

            s_camfore_normalized = t2n(s_camfore)

            s_mask = s_mask.long()
            s_mix_mask = t2n(torch.cat((s_mask, s_pseudo_masks[i].unsqueeze(0)), dim=0))
            s_mask = t2n(s_mask)

            plt.imsave(ospj(vis_path, q_name, f"s_cammask_{s_name}.png"), generate_vis(s_mix_mask, s_vis_image, rate=0.5, color='auto'))    
            plt.imsave(ospj(vis_path, q_name, f"s_camfore_{s_name}.png"), generate_vis(s_camfore_normalized, s_vis_image, color='cam'))

    return s_cams, q_cam, s_masks, s_pseudo_masks.unsqueeze(1), q_pseudo_masks.unsqueeze(1)