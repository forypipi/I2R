import glob
import os
import os.path
from pathlib import Path
from typing import Dict, List, Union
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

        self.num_classes = 1000

        mode = 'train' if self.mode=='train' else 'val'
        self.base_path = self.base_data_root

        with open(f'./lists/{data_set}/splits/{split}.txt', 'r') as f:
            self.categories = f.read().split('\n')[:-1]

        self.sub_list = list(range(1, len(self.categories)))

        print('sub_list: ', self.sub_list)

        exclude_img = ['peregine_falcon/8.jpg']
        self.exclude_img = [os.path.join(self.base_path, name) for name in exclude_img]

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        

    # rewrite len
    def __len__(self):
        return len(self.img_metadata)

    # rewrite getitem
    def __getitem__(self, idx):
        query_name, support_names, subcls = self.sample_episode(idx)
        query_img_name = f'{subcls}_{query_name.split("/")[-1]}'
        s_names = []
        for s_path in support_names:
            s_names.append(f'{subcls}_{os.path.basename(s_path).split(".")[0]}') 

        image, label, s_x, s_y = self.load_frame(query_name, support_names)

        support_image_list_ori = copy.deepcopy(s_x)
        support_label_list_ori = transform_anns(s_y, self.ann_type)[0]
        support_label_list_ori_mask = copy.deepcopy(s_y)

        label_b = np.zeros_like(label)

        raw_image = image.copy()
        raw_label_shape = torch.tensor(label.shape)
        raw_label_b_shape = torch.tensor(label_b.shape)
        _, direct_label, direct_label_b = self.transform_ori(image, label, label_b)

        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform_tri is not None and self.transform is not None:
            image, label, label_b = self.transform_tri(image, label, label_b)   # transform the triple
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k], support_label_list_ori[k])

        s_x = support_image_list[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([support_image_list[i].unsqueeze(0), s_x], 0)
        s_y = support_label_list[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([support_label_list[i].unsqueeze(0), s_y], 0)

        # Return
        if self.mode == 'train':
            batch = {
                'x': image,
                'y_m': label,
                'y_b': label_b,
                's_x': s_x,
                's_y': s_y,
                'cat_idx': subcls,
            }
            return batch
        elif self.mode in ['val', 'test']:
            batch = {
                'x': image,
                'y_m': label,
                'y_b': label_b,
                's_x': s_x,
                's_y': s_y,
                'cat_idx': subcls,
                'raw_label': direct_label,
                'raw_label_b': direct_label_b,
                'raw_label_shape': raw_label_shape,
                'raw_label_b_shape': raw_label_b_shape,
                'q_name': query_img_name,
                's_name': s_names
            }
            return batch
        elif self.mode == 'demo':
            total_image_list = support_image_list_ori.copy()
            total_image_list.append(raw_image)
            batch = {
                'x': image,
                'y_m': label,
                'y_b': label_b,
                's_x': s_x,
                's_y': s_y,
                'cat_idx': subcls,
                'total_image_list': total_image_list,
                'support_label_list_ori': support_label_list_ori,
                'support_label_list_ori_mask': support_label_list_ori_mask,
                'raw_label': direct_label,
                'raw_label_b': direct_label_b,
                'raw_label_shape': raw_label_shape,
                'raw_label_b_shape': raw_label_b_shape,
            }
            return batch
            

    def load_frame(self, query_name, support_names):
        query_img = cv2.imread(query_name, cv2.IMREAD_COLOR)
        support_imgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name and support_name not in self.exclude_img: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob(f'{os.path.join(self.base_path, cat)}/*')])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg' and img_path not in self.exclude_img:
                    img_metadata.append(img_path)
        return img_metadata