r""" PACO-Part few-shot semantic segmentation dataset """
from math import ceil, floor
import os
import pickle

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from detectron2.structures.masks import *

from data.BaseDataset import BaseDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

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
                 box_crop=True,
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
        self.nfolds = 4
        self.nclass = 448
        self.benchmark = 'paco_part'
        self.shot = shot
        self.img_path = os.path.join(data_root, 'PACO-Part', 'coco')
        self.anno_path = os.path.join(data_root, 'PACO-Part', 'paco')
        self.transform = transform
        self.transform_tri = transform_tri
        self.transform_ori = transform_ori
        self.box_crop = box_crop

        self.class_ids_ori, self.cid2img, self.img2anno = self.build_img_metadata_classwise()        
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))
        self.img_metadata = self.build_img_metadata()


    def check(self):


        print(f'<<< checking {self.mode} dataset... <<<')

        exclude_query_name = {'2128_train2017/000000351597.jpg_412185'}

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

        # 检查 batch 是否正常
        for class_sample in sorted(self.class_ids_ori):

            # 按照 query 排序迭代
            queries = sorted(self.cid2img[class_sample], key=lambda q: list(q.values())[0])
            for query in queries:

                query_id, query_name = list(query.keys())[0], list(query.values())[0]
                query_name = '/'.join( query_name.split('/')[-2:])

                query_img = Image.open(os.path.join(self.img_path, query_name)).convert('RGB')
                org_qry_imsize = query_img.size
                query_annos = self.img2anno[query_id]

                query_obj_dict = {}

                for anno in query_annos:
                    if anno['category_id'] == class_sample:
                        obj_id = anno['obj_ann_id']
                        if obj_id not in query_obj_dict:
                            query_obj_dict[obj_id] = {
                                'obj_bbox': [],
                                'segms': []
                            }
                        query_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                        query_obj_dict[obj_id]['segms'].append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...])

                for sel_query_id in list(query_obj_dict.keys()):

                    image_name_full = f'{class_sample}_{query_name}_{sel_query_id}'

                    if image_name_full in exclude_query_name:
                        print(f'Skipping excluded query image {image_name_full}')
                        continue

                    print(f'Checking class {class_sample}, query image {query_name}, object id {sel_query_id}')

                    query_obj_bbox = query_obj_dict[sel_query_id]['obj_bbox'][0]
                    query_part_masks = query_obj_dict[sel_query_id]['segms']
                    query_mask = torch.cat(query_part_masks, dim=0)
                    query_mask = query_mask.sum(0) > 0
                    query_mask = query_mask.numpy()

                    if self.box_crop:
                        new_query_img = np.asarray(query_img)

                        print(f' - Query object bbox: {query_obj_bbox}')

                        new_query_img = new_query_img[floor(query_obj_bbox[1]):floor(query_obj_bbox[1])+ceil(query_obj_bbox[3]), floor(query_obj_bbox[0]):floor(query_obj_bbox[0])+ceil(query_obj_bbox[2])]
                        new_query_img = Image.fromarray(np.uint8(new_query_img))
                        query_mask = query_mask[floor(query_obj_bbox[1]):floor(query_obj_bbox[1])+ceil(query_obj_bbox[3]), floor(query_obj_bbox[0]):floor(query_obj_bbox[0])+ceil(query_obj_bbox[2])]
                    
                    print(f' - after box crop: {new_query_img.size}, mask size: {query_mask.shape}')
                    try:
                        self.transform(new_query_img, query_mask.astype(np.uint8))
                        self.transform_tri(new_query_img, query_mask.astype(np.uint8), query_mask.astype(np.uint8))
                        self.transform_ori(new_query_img, query_mask.astype(np.uint8), query_mask.astype(np.uint8))
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
        return len(self.img_metadata) if self.mode == 'train' else 2500

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        ori_query_img, ori_query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_query_img_size = self.load_frame()

        query_img, query_mask = self.transform(ori_query_img, ori_query_mask.astype(np.uint8))
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.shape[-2:], mode='nearest').squeeze()

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
            raise ValueError('Wrong mode: {}'.format(self.mode))
        

    def build_img_metadata_classwise(self):

        with open(os.path.join(self.anno_path, 'paco_part_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'paco_part_val.pkl'), 'rb') as f:
            test_anno = pickle.load(f)

        # Remove Duplicates
        new_cid2img = {}

        for cid_id in test_anno['cid2img']:
            id_list = []
            if cid_id not in new_cid2img:
                new_cid2img[cid_id] = []
            for img in test_anno['cid2img'][cid_id]:
                img_id = list(img.keys())[0]
                if img_id not in id_list:
                    id_list.append(img_id)
                    new_cid2img[cid_id].append(img)
        test_anno['cid2img'] = new_cid2img

        train_cat_ids = list(train_anno['cid2img'].keys())
        test_cat_ids = [i for i in list(test_anno['cid2img'].keys()) if len(test_anno['cid2img'][i]) > self.shot]
        assert len(train_cat_ids) == self.nclass

        nclass_trn = self.nclass // self.nfolds

        class_ids_val = [train_cat_ids[self.split + self.nfolds * v] for v in range(nclass_trn)]
        class_ids_val = [x for x in class_ids_val if x in test_cat_ids]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.mode == 'train' else class_ids_val
        img_metadata_classwise = train_anno if self.mode == 'train' else test_anno
        cid2img = img_metadata_classwise['cid2img']
        img2anno = img_metadata_classwise['img2anno']

        return class_ids, cid2img, img2anno

    def build_img_metadata(self):
        img_metadata = []
        for k in self.cid2img.keys():
            img_metadata += self.cid2img[k]
        return img_metadata

    def get_mask(self, segm, image_size):

        if isinstance(segm, list):
            # polygon
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


    def load_frame(self):

        exclude_query_name = {'2128_train2017/000000351597.jpg_412185'}

        while True:
            class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]

            query = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
            query_id, query_name = list(query.keys())[0], list(query.values())[0]
            query_name = '/'.join( query_name.split('/')[-2:])
            query_img = Image.open(os.path.join(self.img_path, query_name)).convert('RGB')
            org_qry_imsize = query_img.size
            query_annos = self.img2anno[query_id]

            query_obj_dict = {}

            for anno in query_annos:
                if anno['category_id'] == class_sample:
                    obj_id = anno['obj_ann_id']
                    if obj_id not in query_obj_dict:
                        query_obj_dict[obj_id] = {
                            'obj_bbox': [],
                            'segms': []
                        }
                    query_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                    query_obj_dict[obj_id]['segms'].append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...])

            sel_query_id = np.random.choice(list(query_obj_dict.keys()), 1, replace=False)[0]

            query_obj_bbox = query_obj_dict[sel_query_id]['obj_bbox'][0]
            query_part_masks = query_obj_dict[sel_query_id]['segms']
            query_mask = torch.cat(query_part_masks, dim=0)
            query_mask = query_mask.sum(0) > 0
            query_mask = query_mask.numpy()

            if f'{class_sample}_{query_name}_{sel_query_id}' not in exclude_query_name:
                break

        support_names = []
        support_pre_masks = []
        support_boxes = []
        while True:  # keep sampling support set if query == support

            support = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
            support_id, support_name = list(support.keys())[0], list(support.values())[0]

            support_name = '/'.join(support_name.split('/')[-2:])
            if query_name != support_name:
                support_annos = self.img2anno[support_id]

                support_obj_dict = {}
                for anno in support_annos:
                    if anno['category_id'] == class_sample:
                        obj_id = anno['obj_ann_id']
                        if obj_id not in support_obj_dict:
                            support_obj_dict[obj_id] = {
                                'obj_bbox': [],
                                'segms': []
                            }
                        support_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                        support_obj_dict[obj_id]['segms'].append(anno['segmentation'])

                sel_support_id = np.random.choice(list(support_obj_dict.keys()), 1, replace=False)[0]
                support_obj_bbox = support_obj_dict[sel_support_id]['obj_bbox'][0]
                support_part_masks = support_obj_dict[sel_support_id]['segms']

                if f'{class_sample}_{support_name}_{sel_support_id}' not in exclude_query_name:
                    support_names.append(support_name)
                    support_boxes.append(support_obj_bbox)
                    support_pre_masks.append(support_part_masks)

            if len(support_names) == self.shot:
                break

        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.img_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...])
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0
            support_mask = support_mask.numpy()

            support_masks.append(support_mask)

        if self.box_crop:

            bias = 0
            
            query_img = np.asarray(query_img)

            x, y = query_img.shape[1], query_img.shape[0]

            # Expand the bounding box by bias pixels on each side, ensuring it stays within image bounds
            left = max(floor(query_obj_bbox[0]) - bias, 0)
            top = max(floor(query_obj_bbox[1]) - bias, 0)
            right = min(floor(query_obj_bbox[0]) + ceil(query_obj_bbox[2]) + bias, x)
            bottom = min(floor(query_obj_bbox[1]) + ceil(query_obj_bbox[3]) + bias, y)

            query_img = query_img[top:bottom, left:right]
            query_img = Image.fromarray(np.uint8(query_img))
            org_qry_imsize = query_img.size
            query_mask = query_mask[top:bottom, left:right]

            new_support_imgs = []
            new_support_masks = []

            for sup_img, sup_mask, sup_box in zip(support_imgs, support_masks, support_boxes):

                sup_img = np.asarray(sup_img)

                s_x, s_y = sup_img.shape[1], sup_img.shape[0]
                left = max(floor(sup_box[0]) - bias, 0)
                top = max(floor(sup_box[1]) - bias, 0)
                right = min(floor(sup_box[0]) + ceil(sup_box[2]) + bias, s_x)
                bottom = min(floor(sup_box[1]) + ceil(sup_box[3]) + bias, s_y)

                sup_img = sup_img[top:bottom, left:right]
                sup_img = Image.fromarray(np.uint8(sup_img))

                new_support_imgs.append(sup_img)
                new_support_masks.append(sup_mask[top:bottom, left:right])

            support_imgs = new_support_imgs
            support_masks = new_support_masks

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


if __name__ == '__main__':

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    transform = Compose([
        # A.ToGray(p=0.2),
        # A.Posterize(p=0.2),
        # A.Equalize(p=0.2),
        # A.Sharpen(p=0.2),
        # A.RandomBrightnessContrast(p=0.2),
        # A.Solarize(p=0.2),
        # A.ColorJitter(p=0.2),
        A.Resize(513, 513),
        A.Normalize(mean, std),
        ToTensorV2(),
        ])

    transform_tri = tri_Compose([
        # A.ToGray(p=0.2),
        # A.Posterize(p=0.2),
        # A.Equalize(p=0.2),
        # A.Sharpen(p=0.2),
        # A.RandomBrightnessContrast(p=0.2),
        # A.Solarize(p=0.2),
        # A.ColorJitter(p=0.2),
        A.Resize(513, 513),
        A.Normalize(mean, std),
        ToTensorV2(),
        ], additional_targets={'mask2': 'mask'})

    transform_ori = transform_tri


    # 'x': query_img,
    # 'y_m': query_mask,
    # 'y_b': query_mask,
    # 's_x': support_imgs,
    # 's_y': support_masks,
    # 'cat_idx': torch.tensor(self.class_ids[self.cat_part_name.index(class_sample)]),
    # 'raw_label': direct_label,
    # 'raw_label_b': direct_label_b,
    # 'raw_label_shape': raw_label_shape,
    # 'raw_label_b_shape': raw_label_b_shape,
    # 'q_name': query_img_id,
    # 's_name': support_img_ids
    dataset = SemData(
        data_root='/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets',
        split=1, transform=transform,
        mode='train',
        shot=1, transform_ori=transform_ori,
        transform_tri=transform_tri,)

    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    def save_image_and_mask(img_tensor, mask_tensor, filename_prefix):
        # img_tensor: [3, H, W], mask_tensor: [H, W]
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        mask = mask_tensor.cpu().numpy()
        # Normalize image for visualization
        img_vis = (img - img.min()) / (img.max() - img.min())
        # Mask visualization
        mask_vis = np.zeros_like(img_vis)
        mask_vis[..., 0] = mask  # Red channel for mask
        # Overlay
        overlay = img_vis.copy()
        overlay[mask > 0] = 0.5 * overlay[mask > 0] + 0.5 * mask_vis[mask > 0]
        # Save
        plt.imsave(f'{filename_prefix}_img.png', img_vis)
        plt.imsave(f'{filename_prefix}_mask.png', mask, cmap='gray')
        plt.imsave(f'{filename_prefix}_overlay.png', overlay)

    batch = dataset[0]

    os.makedirs('vis', exist_ok=True)

    # Query
    save_image_and_mask(batch['x'], batch['y_m'], 'vis/query')

    # Support
    for i in range(batch['s_x'].shape[0]):
        save_image_and_mask(batch['s_x'][i], batch['s_y'][i], f'vis/support_{i}')
