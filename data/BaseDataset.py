from abc import abstractmethod
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

from util.get_weak_anns import transform_anns
from util import transform as T
from util import transform_tri as Tri

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):    
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32

    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_dict = {}
    for sub_c in sub_list:
        sub_class_file_dict[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []     

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if tmp_label.sum() >= 2 * 32 * 32:      
                            new_label_class.append(c)
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0],target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_dict[c].append(item)
                    
    print(f"Checking image&label pair {split} list done!")
    return image_label_list, sub_class_file_dict

class BaseDataset(Dataset):
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
    
    # pascal version
    def __init__(self,
                 mode='train',
                 split=None,
                 shot=1,
                 data_root=None,
                 base_data_root=None,
                 ann_type=None,
                 transform=None, 
                 transform_tri=None, 
                 transform_ori=None,
                 *args,
                 **kwargs
                 ):
        
        assert mode in ['train', 'val', 'demo', 'finetune', 'test']

        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root
        self.base_data_root = base_data_root
        self.ann_type = ann_type
        self.transform = transform
        self.transform_tri = transform_tri
        self.transform_ori = transform_ori
      
    def __len__(self):
        if self.mode == 'train':
            return len(self.data_list)
        else:
            return 1000

    def __getitem__(self, index):
        label_class = []
        
        index = index % len(self.data_list)
        
        image_path, label_path = self.data_list[index]
        query_img_name = os.path.basename(image_path).split('.')[0]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_b = cv2.imread(os.path.join(self.base_path, label_path.split('/')[-1]), cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []
        for c in label_class:
            if self.mode in ['val', 'demo', 'finetune', 'test'] and c in self.sub_val_list:
                new_label_class.append(c)
            if self.mode == 'train' and c in self.sub_list:
                new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        class_chosen = label_class[random.randint(1,len(label_class))-1]
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255

        file_class_chosen = self.sub_class_file_dict[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]
                support_image_path = os.path.join(self.data_root, support_image_path)
                support_label_path = os.path.join(self.data_root, support_label_path)
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        if self.mode == 'train':
            subcls = self.sub_list.index(class_chosen)
        else:
            subcls = self.sub_val_list.index(class_chosen)

        s_names = []
        for k in range(self.shot):  
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            s_names.append(os.path.basename(support_label_path).split('.')[0]) 

            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            
            support_label, support_label_mask = transform_anns(support_label, self.ann_type)   # mask/bbox
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot                    
        
        raw_image = image.copy()
        raw_label_shape = torch.tensor(label.shape)
        raw_label_b_shape = torch.tensor(label_b.shape)

        _, direct_label, direct_label_b = self.transform_ori(image, label, label_b)

        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform_tri is not None and self.transform is not None:
            image, label, label_b = self.transform_tri(np.uint8(image), label, label_b)   # transform the triple
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(np.uint8(support_image_list_ori[k]), support_label_list_ori[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

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