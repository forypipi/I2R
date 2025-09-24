r""" Dataloader builder for few-shot semantic segmentation dataset  """
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from util import transform, transform_tri, config
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data import pascal, coco, FSS1000, pascal_part, paco_part, lvis

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

class FSSDatasetModule(LightningDataModule):
    """
    A LightningDataModule for FS-S benchmark
    """
    def __init__(self, args):
        super().__init__()

        self.args = args

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean_255 = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std_255 = [item * value_scale for item in std]

        if self.args.data_set in ['pascal', 'coco']:
            self.train_transform = transform.Compose([
                transform.RandScale([args.scale_min, args.scale_max]),
                transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean_255, ignore_label=args.padding_label),
                transform.RandomGaussianBlur(),
                transform.RandomHorizontalFlip(),
                transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean_255, ignore_label=args.padding_label),
                transform.ToTensor(),
                transform.Normalize(mean=mean_255, std=std_255)])
            
            self.train_transform_tri = transform_tri.Compose([
                transform_tri.RandScale([args.scale_min, args.scale_max]),
                transform_tri.RandRotate([args.rotate_min, args.rotate_max], padding=mean_255, ignore_label=args.padding_label),
                transform_tri.RandomGaussianBlur(),
                transform_tri.RandomHorizontalFlip(),
                transform_tri.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean_255, ignore_label=args.padding_label),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean_255, std=std_255)])

            if args.resized_val:
                self.val_transform = transform.Compose([
                    transform.Resize(size=args.val_size),
                    transform.ToTensor(),
                    transform.Normalize(mean=mean_255, std=std_255)])
                self.val_transform_tri = transform_tri.Compose([
                    transform_tri.Resize(size=args.val_size),
                    transform_tri.ToTensor(),
                    transform_tri.Normalize(mean=mean_255, std=std_255)])

            else:
                self.val_transform = transform.Compose([
                    transform.test_Resize(size=args.val_size),
                    transform.ToTensor(),
                    transform.Normalize(mean=mean_255, std=std_255)])
                self.val_transform_tri = transform_tri.Compose([
                    transform_tri.test_Resize(size=args.val_size),
                    transform_tri.ToTensor(),
                    transform_tri.Normalize(mean=mean_255, std=std_255)])

        elif self.args.data_set in ['FSS1000', 'pascal_part', 'lvis', 'paco_part']:
            self.train_transform = Compose([
                A.ToGray(p=0.2),
                A.Posterize(p=0.2),
                A.Equalize(p=0.2),
                A.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.ColorJitter(p=0.2),
                A.Resize(args.train_h, args.train_w),
                A.Normalize(mean, std),
                ToTensorV2(),
                ])
            
            self.train_transform_tri = tri_Compose([
                A.ToGray(p=0.2),
                A.Posterize(p=0.2),
                A.Equalize(p=0.2),
                A.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.ColorJitter(p=0.2),
                A.Resize(args.train_h, args.train_w),
                A.Normalize(mean, std),
                ToTensorV2(),
                ], additional_targets={'mask2': 'mask'})

            self.val_transform = Compose([
                A.Resize(args.val_size, args.val_size),
                A.Normalize(mean, std),
                ToTensorV2(),
                ])

            self.val_transform_tri = tri_Compose([
                A.Resize(args.val_size, args.val_size),
                A.Normalize(mean, std),
                ToTensorV2(),
                ], additional_targets={'mask2': 'mask'})
            
        else:
            raise ValueError('Unknown dataset: {}'.format(self.args.data_set))


        self.tst_transform = self.val_transform
        self.tst_transform_tri = self.val_transform_tri
        self.transform_ori = self.val_transform_tri

        dataset_dict = {
            'pascal': pascal.SemData,
            'coco': coco.SemData,
            'FSS1000': FSS1000.SemData,
            'pascal_part': pascal_part.SemData,
            'paco_part': paco_part.SemData,
            'lvis': lvis.SemData
        }
        self.dataset = dataset_dict[args.data_set]

    def train_dataloader(self):

        if self.args.data_set == 'FSS1000':
            split = 'trn'
        else:
            split = self.args.split

        train_data = self.dataset(
            split=split,
            fold=self.args.fold,
            shot=self.args.shot, 
            data_root=self.args.data_root, 
            base_data_root=self.args.base_data_root,
            data_list=self.args.train_list,        # not used
            transform=self.train_transform, 
            transform_tri=self.train_transform_tri,
            transform_ori=self.transform_ori,
            mode='train',
            data_set=self.args.data_set, 
            use_split_coco=self.args.use_split_coco)
        
        dataloader = DataLoader(
            train_data,
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=9, 
            drop_last=True,
            pin_memory=True,
            persistent_workers=False
            )
        return dataloader

    def val_dataloader(self):
        # using original image size for validation and batch size is 1

        if self.args.data_set == 'FSS1000':
            split = 'test'
        else:
            split = self.args.split

        val_data = self.dataset(
            split=split,
            fold=self.args.fold,
            shot=self.args.shot, 
            data_root=self.args.data_root, 
            base_data_root=self.args.base_data_root, 
            data_list=self.args.val_list,       # not used
            transform=self.val_transform,
            transform_tri=self.val_transform_tri,
            transform_ori=self.transform_ori,
            mode='val', 
            data_set=self.args.data_set, 
            use_split_coco=self.args.use_split_coco)
           
        dataloader = DataLoader(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False, 
            num_workers=9,
            pin_memory=True,
            persistent_workers=True
            )
        return dataloader

    def test_dataloader(self):
        # using original image size for validation and batch size is 1
        
        if self.args.data_set == 'FSS1000':
            split = 'test'
        else:
            split = self.args.split

        tst_data = self.dataset(
            split=split,
            fold=self.args.fold,
            shot=self.args.shot,
            data_root=self.args.data_root, 
            base_data_root=self.args.base_data_root, 
            data_list=self.args.val_list,          # not used
            transform=self.tst_transform,
            transform_tri=self.tst_transform_tri,
            transform_ori=self.transform_ori,
            mode='test',
            data_set=self.args.data_set, 
            use_split_coco=self.args.use_split_coco)

        dataloader = DataLoader(
            tst_data, 
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=9, 
            pin_memory=True,
            persistent_workers=True
            )
        return dataloader
