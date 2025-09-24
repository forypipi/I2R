import os
from os.path import join
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import json
import cv2
import pycocotools.mask as mask_util
from torchvision import transforms

from data.BaseDataset import BaseDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
                 mode,
                 data_root,
                 transform,
                 transform_tri, 
                 transform_ori,
                 split, 
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

        self.mode = mode
        
        self.mode = 'val' if mode in ['val', 'test'] else 'train'

        self.split_list = ['animals', 'indoor', 'person', 'vehicles']

        if self.mode == 'train':
            self.cat = [self.split_list[s] for s in [0,1,2,3] if s != split]
        else:
            self.cat = [self.split_list[split]]
        # self.nfolds = 4
        self.benchmark = 'pascal_part'
        self.shot = shot
        self.transform = transform
        self.transform_tri = transform_tri
        self.transform_ori = transform_ori
        self.box_crop = box_crop

        self.json_file = os.path.join(data_root, 'Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json')
        self.img_file = os.path.join(data_root, 'Pascal-Part/VOCdevkit/VOC2010/JPEGImages/{}.jpg')
        self.anno_file = os.path.join(data_root, 'Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes/{}.json')
        js = json.load(open(self.json_file, 'r'))

        self.cat_annos = []

        for c in self.cat:
            self.cat_annos.append(js[c])

        cat_part_name = []      # 'bird+FACE', 'bird+LEGS', 'bird+TAIL', 'bird+WING', ...
        cat_part_id = []        # 0, 1, 2, 3, ...

        # 统计“类别-部分”的组合，并为每个组合分配一个新的ID
        new_id = 0
        for i, cat_anno in enumerate(self.cat_annos):
            for obj in cat_anno['object']:        # 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep'
                for part in cat_anno['object'][obj]['part']:  # 'FACE', 'LEGS', 'TAIL', 'WING', ...
                    if len(cat_anno['object'][obj]['part'][part]['train']) > 0 and \
                            len(cat_anno['object'][obj]['part'][part]['val']) > 0:
                        if obj + '+' + part == 'aeroplane+TAIL':
                            continue
                        
                        cat_part_name.append(f'{i}+{obj}+{part}')
                        cat_part_id.append(new_id)
                        new_id += 1

        self.cat_part_name = cat_part_name
        self.class_ids = self.cat_part_id = cat_part_id
        self.nclass = len(cat_part_id)

        split_info = []
        for i in self.cat_part_name:
            fold_id, obj_n, part_n = i.split('+')
            split_info.append(f'{self.cat[int(fold_id)]}+{obj_n}+{part_n}')
        print(self.mode, self.split, split_info)

        self.img_metadata = self.build_img_metadata()       # mode (train/val)对应的图像编号，2008_000472, 2008_000536, 2008_000537, ...

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_metadata)
        else:
            return min(len(self.img_metadata), 2500)

    def build_img_metadata(self):

        img_metadata = []
        for cat_anno in self.cat_annos:
            for obj in cat_anno['object']:
                for part in cat_anno['object'][obj]['part']:
                    img_metadata.extend(cat_anno['object'][obj]['part'][part][self.mode])

        return img_metadata

    def __getitem__(self, idx):

        idx %= len(self.class_ids)      # 类别id
        ori_query_img, ori_query_mask, support_imgs, support_masks, query_img_id, support_img_ids, class_sample, org_query_img_size = self.sample_episode(idx)

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

        _, direct_label, direct_label_b = self.transform_ori(ori_query_img, ori_query_mask.astype(np.uint8), ori_query_mask.astype(np.uint8))     # image, label_m, label_b

        # Return
        if self.mode == 'train':
            batch = {
                'x': query_img,
                'y_m': query_mask,
                'y_b': query_mask,
                's_x': support_imgs,
                's_y': support_masks,
                'cat_idx': torch.tensor(self.class_ids[self.cat_part_name.index(class_sample)]),
            }
            return batch
        elif self.mode in ['val', 'test']:
            batch = {
                'x': query_img,
                'y_m': query_mask,
                'y_b': query_mask,
                's_x': support_imgs,
                's_y': support_masks,
                'cat_idx': torch.tensor(self.class_ids[self.cat_part_name.index(class_sample)]),
                'raw_label': direct_label,
                'raw_label_b': direct_label_b,
                'raw_label_shape': raw_label_shape,
                'raw_label_b_shape': raw_label_b_shape,
                'q_name': query_img_id,
                's_name': support_img_ids
            }
            return batch
    

    def sample_episode(self, idx):

        class_sample, class_sample_id = self.cat_part_name[idx], self.class_ids[idx]
        fold_id, obj_n, part_n = class_sample.split('+')     # 类别和部分
        fold_id = int(fold_id)

        # query
        while True:
            query_img_id = np.random.choice(self.cat_annos[fold_id]['object'][obj_n]['part'][part_n][self.mode], 1, replace=False)[0]
            # data_root/Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes/{query_img_id}.json
            anno = json.load(open(self.anno_file.format(query_img_id), 'r'))    

            sel_obj_in_img = []
            for o in anno['object']:
                if o['name'] == obj_n:      # 同一张图可能有多个类别，仅选择query相关的类别
                    sel_obj_in_img.append(o)    

            assert len(sel_obj_in_img) > 0

            sel_obj = np.random.choice(sel_obj_in_img, 1, replace=False)[0]     # 随机选择一个类别

            sel_parts = []
            for p in sel_obj['parts']:
                if p['name'] == part_n:
                    sel_parts.append(p)

            if not sel_parts:   # 如果选择的类别没有对应的部分，重新选择图像
                continue

            part_masks = []
            for sel_part in sel_parts:
                part_masks.extend(sel_part['mask'])     # 可能一个部分有多个mask，包含size、counts两个字段
            for mask in part_masks:
                mask['counts'] = mask['counts'].encode("ascii")     # 字符串转ascii转mask
            part_mask = mask_util.decode(part_masks)
            part_mask = part_mask.sum(-1) > 0

            if part_mask.size > 0:
                break

        query_img = Image.open(self.img_file.format(query_img_id)).convert('RGB')
        org_qry_imsize = query_img.size
        query_mask = part_mask
        query_obj_box = [int(sel_obj['bndbox'][b]) for b in sel_obj['bndbox']]  # xyxy，定位每一个类别的位置框

        support_img_ids = []
        support_masks = []
        support_boxes = []

        while True:  # keep sampling support set if query == support

            while True:
                support_img_id = \
                np.random.choice(self.cat_annos[fold_id]['object'][obj_n]['part'][part_n][self.mode], 1, replace=False)[0]
                if support_img_id == query_img_id or support_img_id in support_img_ids: continue

                anno = json.load(open(self.anno_file.format(support_img_id), 'r'))

                sel_obj_in_img = []
                for o in anno['object']:
                    if o['name'] == obj_n:
                        sel_obj_in_img.append(o)

                assert len(sel_obj_in_img) > 0

                sel_obj = np.random.choice(sel_obj_in_img, 1, replace=False)[0]

                sel_parts = []
                for p in sel_obj['parts']:
                    if p['name'] == part_n:
                        sel_parts.append(p)

                if not sel_parts:
                    continue

                part_masks = []
                for sel_part in sel_parts:
                    part_masks.extend(sel_part['mask'])
                for mask in part_masks:
                    mask['counts'] = mask['counts'].encode("ascii")
                part_mask = mask_util.decode(part_masks)
                part_mask = part_mask.sum(-1) > 0

                if part_mask.size > 0:
                    break

            support_img_ids.append(support_img_id)
            support_masks.append(part_mask)
            support_boxes.append([int(sel_obj['bndbox'][b]) for b in sel_obj['bndbox']])  # xyxy
            if len(support_img_ids) == self.shot: break

        support_imgs = [Image.open(self.img_file.format(sup_img_id)).convert('RGB') for sup_img_id in support_img_ids]

        # 仅保留mask相关区域
        if self.box_crop:
            query_img = np.asarray(query_img)
            query_img = query_img[query_obj_box[1]:query_obj_box[3], query_obj_box[0]:query_obj_box[2]]
            query_img = Image.fromarray(np.uint8(query_img))
            org_qry_imsize = query_img.size
            query_mask = query_mask[query_obj_box[1]:query_obj_box[3], query_obj_box[0]:query_obj_box[2]]

            new_support_imgs = []
            new_support_masks = []

            for sup_img, sup_mask, sup_box in zip(support_imgs, support_masks, support_boxes):
                sup_img = np.asarray(sup_img)
                sup_img = sup_img[sup_box[1]:sup_box[3], sup_box[0]:sup_box[2]]
                sup_img = Image.fromarray(np.uint8(sup_img))

                new_support_imgs.append(sup_img)
                new_support_masks.append(sup_mask[sup_box[1]:sup_box[3], sup_box[0]:sup_box[2]])

            support_imgs = new_support_imgs
            support_masks = new_support_masks

        return query_img, query_mask, support_imgs, support_masks, query_img_id, support_img_ids, class_sample, org_qry_imsize


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
        split=3, transform=transform,
        mode='val',
        shot=5, transform_ori=transform_ori,
        transform_tri=transform_tri,)
    
    batch = dataset.__getitem__(0)
    query_img = batch['x']
    query_mask = batch['y_m']
    support_imgs = batch['s_x']
    support_masks = batch['s_y']
    query_img_id = batch['q_name']
    support_img_ids = batch['s_name']
    class_sample = batch['cat_idx']
    org_qry_imsize = batch['raw_label_shape']
    
    print('query img:', query_img.shape)
    print('query mask:', query_mask.shape)
    print('query unique:', np.unique(np.array(query_mask)))
    print('support imgs:', [img.shape for img in support_imgs])
    print('support masks:', [mask.shape for mask in support_masks])
    print('query img id:', query_img_id)
    print('support img ids:', support_img_ids)
    print('class sample:', class_sample)
    print('original query image size:', org_qry_imsize)

    print(len(support_imgs), len(support_masks))


    # 可视化，保存
    for i in range(len(support_imgs)):
        print(np.unique(np.array(support_masks[i])))
        print(np.sum(support_masks[i].cpu().numpy()))

        import matplotlib.pyplot as plt

        def visualize_query(query_img, query_mask, support_img, support_mask):
            # query_img: torch.Tensor (C, H, W)
            # query_mask: torch.Tensor (H, W)
            img_np = query_img.permute(1, 2, 0).cpu().numpy()
            mask_np = query_mask.cpu().numpy()

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.title('Query Image')
            plt.imshow(img_np)
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.title('Query Mask')
            plt.imshow(mask_np, cmap='gray', alpha=1.0)
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.title('Overlay')
            plt.imshow(img_np)
            mask_overlay = np.zeros_like(img_np)
            mask_overlay[..., 0] = mask_np * 255  # 红色通道
            plt.imshow(mask_overlay, alpha=0.5)
            plt.axis('off')

            # Support Image and Mask
            support_img_np = support_img.permute(1, 2, 0).cpu().numpy()
            support_mask_np = support_mask.cpu().numpy()    

            plt.subplot(2, 3, 4)
            plt.title('Support Image')
            plt.imshow(support_img_np)
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.title('Support Mask')
            plt.imshow(support_mask_np, cmap='gray', alpha=1.0)
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.title('Overlay')
            plt.imshow(support_img_np)
            support_mask_overlay = np.zeros_like(support_img_np)
            support_mask_overlay[..., 0] = support_mask_np * 255  # 红色通道
            plt.imshow(support_mask_overlay, alpha=0.5)
            plt.axis('off')

            plt.savefig(f'q_{query_img_id}_s_{support_img_ids[i]}_class_{class_sample}.png')

            plt.show()

        visualize_query(query_img, query_mask, support_imgs[i], support_masks[i])