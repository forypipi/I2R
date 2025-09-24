r""" LVIS-92i few-shot semantic segmentation dataset """
import os
import pickle
from datetime import datetime

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import cv2

from detectron2.structures.masks import *
import pycocotools.mask as mask_util

from data.BaseDataset import BaseDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

class Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)
    
    def __call__(self, image, mask):
        augmented = super().__call__(image=np.array(image), mask=np.array(mask))
        return augmented['image'], augmented['mask']

class tri_Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)
    
    def __call__(self, image, mask1, mask2):
        augmented = super().__call__(image=np.array(image), mask=np.array(mask1), mask2=np.array(mask2))
        return augmented['image'], augmented['mask'], augmented['mask2']
    

class SemData(BaseDataset):
    def __init__(self, 
                 data_root, 
                 split, 
                 transform, 
                 transform_tri, 
                 transform_ori,
                 mode, 
                 shot,
                 base_data_root=None,
                 ann_type='mask',
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
        
        self.mode = 'val' if mode in ['val', 'test'] else 'train'
        self.split = split
        self.nfolds = 10
        self.benchmark = 'lvis'
        self.shot = shot
        self.anno_path = os.path.join(data_root, "LVIS")
        self.base_path = os.path.join(data_root, "LVIS", 'coco')
        self.transform = transform
        self.transform_tri = transform_tri
        self.transform_ori = transform_ori

        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        self.img_metadata = self.build_img_metadata()

    def check(self):

        print(f'<<< checking {self.mode} dataset... <<<')

        # 记录原始 p
        orig_p = {}
        for tf_name in ['transform', 'transform_tri', 'transform_ori']:
            tf = getattr(self, tf_name, None)
            if tf is not None and hasattr(tf, 'transforms'):
                for t in tf.transforms:
                    if hasattr(t, 'p'):
                        print(f'Found transform with p in {tf_name}: {t}, original p: {t.p}')
                        orig_p[(tf_name, id(t))] = t.p
                        t.p = 1

        exclude_query_name = {'train2017/000000483162.jpg', 'train2017/000000479261.jpg'}

        for class_sample in sorted(self.class_ids_ori):

            for query_name in sorted(self.img_metadata_classwise[class_sample].keys()):

                if query_name in exclude_query_name:
                    print(f'Skipping excluded query image {query_name} with class {class_sample}')
                    continue

                query_info = self.img_metadata_classwise[class_sample][query_name]
                query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
                org_qry_imsize = query_img.size
                query_annos = query_info['annotations']
                segms = []

                for anno in query_annos:
                    segms.append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...].float())
                query_mask = torch.cat(segms, dim=0)
                query_mask = query_mask.sum(0) > 0
                query_mask = query_mask.numpy()

                print(f'Checking class {class_sample}, query image {query_name}')

                try:
                    query_img2, _ = self.transform(query_img, query_mask.astype(np.uint8))
                    query_img3, query_img3_mask, _ = self.transform_tri(query_img, query_mask.astype(np.uint8), query_mask.astype(np.uint8))
                    self.transform_ori(query_img, query_mask.astype(np.uint8), query_mask.astype(np.uint8))

                    # 分channel计算 max min mean std
                    if isinstance(query_img2, torch.Tensor):
                        # shape: [C, H, W]
                        for c in range(query_img2.shape[0]):
                            channel = query_img2[c]
                            print(f'    query 2 channel {c}: {channel.dtype}, max={channel.max().item()}, min={channel.min().item()}, mean={channel.mean().item()}, std={channel.std().item()}')
                    if isinstance(query_img3, torch.Tensor):
                        for c in range(query_img3.shape[0]):
                            channel = query_img3[c]
                            print(f'    query 3 channel {c}: {channel.dtype}, max={channel.max().item()}, min={channel.min().item()}, mean={channel.mean().item()}, std={channel.std().item()}')

                    print(f'  query 3 mask: {query_img3_mask.dtype} {query_img3_mask.max()}, {query_img3_mask.min()}')
                except Exception as e:
                    print(f'Error processing query image {query_name} with class {class_sample}: {e}')
                    raise e

        # 恢复原始 p
        for tf_name in ['transform', 'transform_tri', 'transform_ori']:
            tf = getattr(self, tf_name, None)
            if tf is not None and hasattr(tf, 'transforms'):
                for t in tf.transforms:
                    if hasattr(t, 'p') and (tf_name, id(t)) in orig_p:
                        t.p = orig_p[(tf_name, id(t))]
        
        raise Exception('Check finished.')



    def __len__(self):
        return len(self.img_metadata) if self.mode == 'train' else 2300

    def __getitem__(self, idx):
        idx %= len(self.class_ids)

        ori_query_img, ori_query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_query_img_size = self.load_frame(idx)

        query_img, query_mask = self.transform(ori_query_img, ori_query_mask.astype(np.uint8))
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_img_list = [[] for _ in range(self.shot)]
        support_mask_list = [[] for _ in range(self.shot)]
        if self.transform_tri is not None and self.transform is not None:
            query_img, query_mask, _ = self.transform_tri(ori_query_img, ori_query_mask.astype(np.uint8), ori_query_mask.astype(np.uint8))   # transform the triple
            for k in range(self.shot):
                supp_img, supp_mask = self.transform(support_imgs[k], support_masks[k].astype(np.uint8))
                support_img_list[k] = supp_img
                support_mask_list[k] = F.interpolate(supp_mask.unsqueeze(0).unsqueeze(0).float(), supp_img.shape[-2:], mode='nearest').squeeze()

        support_imgs, support_masks = torch.stack(support_img_list), torch.stack(support_mask_list)

        raw_label_shape = torch.tensor(org_query_img_size)
        raw_label_b_shape = torch.tensor(org_query_img_size)

        _, direct_label, direct_label_b = self.transform_ori(ori_query_img, ori_query_mask.astype(np.uint8), ori_query_mask.astype(np.uint8))   # image, label_m, label_b

        query_name = query_name.replace('/', '-')
        support_names = [name.replace('/', '-') for name in support_names]

        # Return
        if self.mode == 'train':
            batch = {
                'x': query_img,
                'y_m': query_mask,
                'y_b': query_mask,
                's_x': support_imgs,
                's_y': support_masks,
                'cat_idx': torch.tensor(self.class_ids_c[class_sample]),
                'q_name': query_name,
                's_name': support_names
            }
            return batch
        elif self.mode in ['val', 'test']:
            batch = {
                'x': query_img,
                'y_m': query_mask,
                'y_b': query_mask,
                's_x': support_imgs,
                's_y': support_masks,
                'cat_idx': torch.tensor(self.class_ids_c[class_sample]),
                'raw_label': direct_label,
                'raw_label_b': direct_label_b,
                'raw_label_shape': raw_label_shape,
                'raw_label_b_shape': raw_label_b_shape,
                'q_name': query_name,
                's_name': support_names
            }
            return batch
        else:
            raise ValueError('Wrong mode! Only "train", "val" and "test" are available, but got {}.'.format(self.mode))


    def build_img_metadata_classwise(self):

        with open(os.path.join(self.anno_path, 'lvis_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        train_cat_ids = [i for i in list(train_anno.keys()) if len(train_anno[i]) > self.shot]
        val_cat_ids = [i for i in list(val_anno.keys()) if len(val_anno[i]) > self.shot]

        trn_nclass = len(train_cat_ids)
        val_nclass = len(val_cat_ids)

        nclass_val_spilt = val_nclass // self.nfolds

        class_ids_val = [val_cat_ids[self.split + self.nfolds * v] for v in range(nclass_val_spilt)]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.mode == 'train' else class_ids_val
        nclass = trn_nclass if self.mode == 'train' else val_nclass
        img_metadata_classwise = train_anno if self.mode == 'train' else val_anno

        return nclass, class_ids, img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size):

        if isinstance(segm, list):
            # polygon
            # polygons = [np.asarray(p).reshape(-1, 2)[:,::-1] for p in segm]
            # polygons = [p.reshape(-1) for p in polygons]
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            # COCO RLE
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)

    def load_frame(self, idx):


        exclude_name = {'train2017/000000483162.jpg', 'train2017/000000479261.jpg'}

        class_sample = self.class_ids_ori[idx]

        if len(list(self.img_metadata_classwise[class_sample].keys())) < self.shot + 1:
            raise ValueError(f'Class {class_sample} does not have enough images to sample {self.shot} shots and 1 query.')



        # class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]

        # Sample query_name until it's not in exclude_name
        while True:
            query_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
            if query_name not in exclude_name:
                break

        query_info = self.img_metadata_classwise[class_sample][query_name]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = query_info['annotations']
        segms = []

        for anno in query_annos:
            segms.append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...].float())
        query_mask = torch.cat(segms, dim=0)
        query_mask = query_mask.sum(0) > 0

        query_mask = query_mask.numpy()

        support_names = []
        support_pre_masks = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]

            if support_name in exclude_name:
                continue

            if query_name != support_name:
                support_names.append(support_name)
                support_info = self.img_metadata_classwise[class_sample][support_name]
                support_annos = support_info['annotations']

                support_segms = []
                for anno in support_annos:
                    support_segms.append(anno['segmentation'])
                support_pre_masks.append(support_segms)

            if len(support_names) == self.shot:
                break


        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...].float())
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0

            support_mask = support_mask.numpy()

            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize



if __name__ == '__main__':

    transform = Compose([
        A.Resize(513, 513),
        ToTensorV2(),
        ])

    transform_tri = tri_Compose([
        A.Resize(513, 513),
        ToTensorV2(),
        ], additional_targets={'mask2': 'mask'})

    transform_ori = transform_tri


    # # 'x': query_img,
    # # 'y_m': query_mask,
    # # 'y_b': query_mask,
    # # 's_x': support_imgs,
    # # 's_y': support_masks,
    # # 'cat_idx': torch.tensor(self.class_ids[self.cat_part_name.index(class_sample)]),
    # # 'raw_label': direct_label,
    # # 'raw_label_b': direct_label_b,
    # # 'raw_label_shape': raw_label_shape,
    # # 'raw_label_b_shape': raw_label_b_shape,
    # # 'q_name': query_img_id,
    # # 's_name': support_img_ids

    total_trn = 0
    total_val = 0
    for split in range(10):
        print(f"Processing split {split}")
        dataset = SemData(
            data_root='/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets',
            split=split,
            transform=transform,
            mode='train',
            shot=5,
            transform_ori=transform_ori,
            transform_tri=transform_tri,
        )
        print(f"[TRN] Split {split} class length: {len(dataset.class_ids)}")
        total_trn += len(dataset.class_ids)

        dataset = SemData(
            data_root='/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets',
            split=split,
            transform=transform,
            mode='val',
            shot=5,
            transform_ori=transform_ori,
            transform_tri=transform_tri,
        )
        print(f"[VAL] Split {split} class length: {len(dataset.class_ids)}")
        total_val += len(dataset.class_ids)

    print(f"total_trn unique classes across all splits: {total_trn}")
    print(f"total_val unique classes across all splits: {total_val}")

    # dataset = SemData(
    #     data_root='/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets',
    #     split=3, transform=transform,
    #     mode='train',
    #     shot=1, transform_ori=transform_ori,
    #     transform_tri=transform_tri,)
    
