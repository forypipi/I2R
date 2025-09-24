import os
import os.path
from pathlib import Path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

from data.BaseDataset import BaseDataset, make_dataset
from util.get_weak_anns import transform_anns
from typing import Dict, List, Union
from util import transform as T
from util import transform_tri as Tri

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

class SemData(BaseDataset):
    mode: str       # train/val/demo/finetune/test, uselly from input args
    split: Union[int, str]  # 0/1/2/3/trn/val (last 2 for fss1000), uselly from input args
    shot: int       # 1/5/10, uselly from input args
    data_root: str              # root path for dataset, uselly from input args, like /share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets/VOC2012
    ann_type: str               # mask/bbox, uselly from input args
    sub_list: list              # trn class id list, use for training, like list(range(1, 16))
    sub_val_list: list          # val and test class id list, use for training, like list(range(1, 16))
    base_path: str              # root for base class annotation, /share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets/VOC2012/base_annotation/trn/0
    sub_class_file_dict: Dict[int, List[tuple]]     # use for choose supp image-mask pair, like {1: [(img1, mask1), (img2, mask2)], 2: [(img3, mask3), ...], ...}
    data_list: list             # use for choose query image-mask pair, like [(img1, mask1), (img2, mask2), ...]
    transform: T.Compose        # transforms for supp image-mask pair, uselly from input args
    transform_tri: Tri.Compose  # transforms for query image-mask-basemask pair, uselly from input args
    transform_ori: Union[T.Compose, None]   # same as self.val_transform_tri, uselly from input args

    def __init__(self, 
                 split=3, 
                 shot=1, 
                 data_root=None, 
                 base_data_root=None, 
                 data_set=None, 
                 transform=None, 
                 transform_tri=None, 
                 mode='train', 
                 ann_type='mask',
                 transform_ori=None,
                 *args,
                 **kwargs
                 ):

        super().__init__(
            mode=mode,
            split=split,
            shot=shot,
            data_root=data_root,
            base_data_root=base_data_root,
            ann_type=ann_type,
            transform=transform, 
            transform_ori=transform_ori, 
            transform_tri=transform_tri
            )

        self.num_classes = 80

        if kwargs['use_split_coco']:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if self.split == 3:
                self.sub_val_list = list(range(3, 80, 4))
            elif self.split == 2:
                self.sub_val_list = list(range(3, 80, 4))
            elif self.split == 1:
                self.sub_val_list = list(range(2, 79, 4))
            elif self.split == 0:
                self.sub_val_list = list(range(1, 78, 4))
        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if self.split == 3:
                self.sub_val_list = list(range(61, 81))
            elif self.split == 2:
                self.sub_val_list = list(range(41, 61))
            elif self.split == 1:
                self.sub_val_list = list(range(21, 41))
            elif self.split == 0:
                self.sub_val_list = list(range(1, 21))           

        self.sub_list = list(set(class_list) - set(self.sub_val_list))                    

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        mode = 'train' if self.mode=='train' else 'val'
        self.base_path = os.path.join(self.base_data_root, mode, str(self.split))

        fss_list_root = f'./lists/{data_set}/fss_list/{mode}/'
        fss_data_list_path = fss_list_root + f'data_list_{split}.txt'
        fss_sub_class_file_list_path = fss_list_root + f'sub_class_file_list_{split}.txt'

        # Read FSS Data
        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            img, mask = line.split()
            img = os.path.join(self.data_root, img)
            mask = os.path.join(self.data_root, mask)
            self.data_list.append((img, mask))

        with open(fss_sub_class_file_list_path, 'r') as f:
            f_str = f.read()
        self.sub_class_file_dict = eval(f_str)