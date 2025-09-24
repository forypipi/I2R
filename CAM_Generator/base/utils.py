import os
from pathlib import Path
from typing import List, Union
import cv2
from einops import rearrange, repeat
import numpy as np
import torch
from ..attribution import CAMWrapper
from ..attribution.utils import normalize_saliency

from pydensecrf import densecrf
from pydensecrf.utils import unary_from_softmax
import PIL.Image as Image

def generate_vis(prob:np.ndarray, img, color="cam", rate=0.5):
    '''
    All the input should be np.ndarray for HW, tensor for 2HW(channel=2, mask+pseudo mask)
    img should be 0-255 uint8
    color: r for red, b for blue, g for green, cam for cam map, auto for 2HW(channel=2)
    rate is img rate
    '''

    prob[prob<=0] = 0
    colorlist = []
    colorlist.append(color_pro(prob, img=img, mode='hwc', color=color, rate=rate))
    CAM = np.array(colorlist)/255.0

    return CAM[0, :, :, :]

def color_pro(prob: np.ndarray, img=None, mode='hwc', color='cam', rate=0.5):
    
    if len(prob.shape) == 2:
        H, W = prob.shape

        pro_255 = (prob*255).astype(np.uint8)
        mask = np.expand_dims(pro_255, axis=2).repeat(3, axis=2)

        if color == "r":
            mask[:,:,[0,1]] = 0
        elif color == "b":
            mask[:,:,[1,2]] = 0
        elif color == "g":
            mask[:,:,[0,2]] = 0
        elif color == 'cam':
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        else:
            raise Exception("passing wrong color args, should in 'rgb' or is 'cam'")
        
    elif len(prob.shape) == 3:
        C, H, W = prob.shape
        if C != 2:
            raise Exception(f"prob should in shape [2,H,W], get {C} channel, please check the first element of 'generate_vis' input (prob).")
        mask, pseudo_mask = prob[0,:,:], prob[1,:,:]
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        color_mask[np.logical_and(mask, pseudo_mask)] = torch.tensor([0, 0, 255])   # red for both foreground
        color_mask[np.logical_and(mask, np.logical_not(pseudo_mask))] = np.array([0, 255, 0])   # green for cam- & mask+
        color_mask[np.logical_and(np.logical_not(mask), pseudo_mask)] = np.array([255, 0, 0])   # blue for cam+ & mask-
        mask = color_mask
    else:
        raise Exception(
            f"passing wrong prob args, prob shape {prob.shape}, should in [H,W] shape for only mask/pseudo mask/cam, [2,H,W] shape for stacked mask&pseudo mask, but get shape {prob.shape}")
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    if img is not None:
        if mode == 'hwc':
            assert img.shape[0] == H and img.shape[1] == W, f"{prob.shape}, {img.shape}"
            if color == 'cam':
                mask = cv2.addWeighted(img, rate, mask, 1-rate, 0)
            else:
                mask = cv2.addWeighted(img, 1, mask, 1-rate, 0)

        elif mode == 'chw':
            assert img.shape[1] == H and img.shape[2] == W, f"{prob.shape}, {img.shape}"
            img = np.transpose(img,(1,2,0))
            if color == 'cam':
                mask = cv2.addWeighted(img, rate, mask, 1-rate, 0)
            else:
                mask = cv2.addWeighted(img, 1, mask, 1-rate, 0)
            mask = np.transpose(mask,(2,0,1))
    else:
        if mode == 'chw':
            mask = np.transpose(mask,(2,0,1))	
    return mask

def np_normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float32)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

def torch_minmaxnormalize(cams: torch.Tensor, mask: torch.Tensor):
    """
    cams: [b,h,w]
    mask: [b,h,w]
    return [b,h,w]
    """

    if torch.isnan(cams).any() or cams.min() == cams.max():
        return torch.zeros_like(cams, dtype=cams.dtype, device=cams.device)
    

    b, h, w = cams.size()
    tmp_cams = cams.view(b, -1)
    max_cams = tmp_cams.clone().detach()
    min_cams = tmp_cams.clone().detach()
    tmp_mask = mask.view(b, -1)

    max_cams[tmp_mask==0] = -torch.inf
    masked_max = max_cams.max(dim=1, keepdim=True)[0]       # 0 for value, 1 fo index

    min_cams[tmp_mask==0] = torch.inf
    masked_min = min_cams.min(dim=1, keepdim=True)[0]

    tmp_cams = (tmp_cams - masked_min) / (masked_max - masked_min + 1e-5)
    tmp_cams = tmp_cams.view(b, h, w)
    tmp_cams = tmp_cams * mask
    tmp_cams = torch.nan_to_num(tmp_cams, nan=0., posinf=0., neginf=0.)
    return tmp_cams

def torch_zscorenormalize(cams: torch.Tensor):

    if torch.isnan(cams).any() or cams.min() == cams.max():
        return torch.zeros_like(cams, dtype=cams.dtype, device=cams.device)
    
    b, h, w = cams.size()
    cams = cams.view(b, -1)
    mean, std = cams.mean(), cams.std()
    cams = (cams - mean) / std
    cams = cams.view(b, h, w)
    return cams

def PseudoMaskGeneration(forecams: torch.Tensor, threshold: Union[float, str, torch.Tensor], **kwargs):
    '''
    :params: forecams: foreground cam in shape [b,h,w]
    :params: threshold: 'auto' or float in 0~1
    :params: kwargs: when threshold is 'auto', kwargs includes cam_net(CAMWrapper), img(torch.Tensor, in shape [b,c,h,w]), img_fore_cls(torch.Tensor, in shape [b,1]), target_layer(str); threshold is 'adaptive', kwargs include s_mask in shape [b,1,h,w]
    
    :return: return mask in shape [b,h,w]
    '''

    if threshold == "adaptive":

        s_mask: torch.Tensor = kwargs['s_mask']
        length = 50
        iou_tensor = torch.zeros((length+1, s_mask.size(dim=0)), device=s_mask.device)
        for i in range(length+1):

            threshold = i / length
            cam_mask = torch.zeros_like(forecams, device=forecams.device, dtype=torch.bool)
            cam_mask[forecams > threshold] = True
            gt_mask = rearrange(s_mask.bool(), 'b c h w -> (b c) h w')
            intersection = (cam_mask & gt_mask).float().sum(dim=[1, 2])
            union = (cam_mask | gt_mask).float().sum(dim=[1, 2])
            # 计算IoU
            iou_tensor[i] = intersection / (union + 1e-6)
        threshold_index = torch.argmax(iou_tensor, dim=0)
        threshold = threshold_index / length

        s_threshold = repeat(threshold, 'b -> b h w', h=1, w=1).expand_as(forecams)
        mask = torch.zeros_like(forecams, device=forecams.device, dtype=forecams.dtype)
        mask[forecams > s_threshold] = 1
    
    elif isinstance(threshold, torch.Tensor):
        assert threshold.size(0)==forecams.size(0), f"should have torch.Tensor threshold in shape ({forecams.size(0)},), but get {threshold.shape}"
        q_threshold = repeat(threshold, 'b -> b h w', h=1, w=1).expand_as(forecams)
        mask = torch.zeros_like(forecams, device=forecams.device, dtype=forecams.dtype)
        mask[forecams > q_threshold] = 1

    elif 0 <= eval(threshold) <= 1:
        mask = torch.zeros_like(forecams, device=forecams.device, dtype=forecams.dtype)
        mask[forecams > eval(threshold)] = 1
    else:
        raise Exception(f"args threshold for cam should be 'auto' or 'adaptive' or float between 0 and 1")
    
    return mask, threshold

def t2n(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)

