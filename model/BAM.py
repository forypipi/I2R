from datetime import datetime
import math
import os
from pathlib import Path
from typing import Dict, List, OrderedDict, Union
from os.path import join as ospj

from einops import rearrange, repeat
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


import numpy as np
import time


from model.ASPP import ASPP
from model.SA import SA
from model.PPM import PPM
from model.PSPNet import OneModel as PSPNet
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs, BNC2BCHW
from data.dataset import FSSDatasetModule
from pytorch_optimizer import ScheduleFreeAdamW
from torch.amp import autocast

import pytorch_lightning as pl

from CAM_Generator.CAM_generator import Generator_CAM
from CAM_Generator.base.utils import generate_vis, t2n, torch_minmaxnormalize, torch_zscorenormalize
from CAM_Generator.attribution import CAMWrapper, GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, LayerCAM, EigenCAM, EigenGradCAM, HiResCAM
from model.Decoder import CrossAttn

def Weighted_GAP(supp_feat: torch.Tensor, mask: torch.Tensor):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.adaptive_avg_pool2d(mask, 1) * feat_h * feat_w + 1e-5
    supp_feat = F.adaptive_avg_pool2d(supp_feat, 1) * feat_h * feat_w / area
    return supp_feat.squeeze(-1)

def get_gram_matrix(fea: torch.Tensor):
    b, c, h, w = fea.shape        
    fea = rearrange(fea, 'b c h w -> b c (h w)')    # C*N
    fea_T = rearrange(fea, 'b c d -> b d c')        # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram

def norm_cam_mask(cam_mask: torch.Tensor):
    # 对cam_mask，忽略其值最小的5%像素点重新进行01norm, 37*37=1369, 5%约为68
    cam_mask_flat = cam_mask.view(cam_mask.size(0), -1)
    k = int(cam_mask_flat.size(1) * 0.05)
    topk_vals, _ = torch.topk(cam_mask_flat, cam_mask_flat.size(1) - k, dim=1)
    min_val = topk_vals.min(dim=1, keepdim=True)[0]
    max_val = topk_vals.max(dim=1, keepdim=True)[0]
    cam_mask_norm = (cam_mask_flat - min_val) / (max_val - min_val + 1e-6)
    cam_mask_norm = torch.clamp(cam_mask_norm, 0, 1)
    cam_mask = cam_mask_norm.view_as(cam_mask)
    return cam_mask

class OneModel(pl.LightningModule):
    best_miou: int
    best_FBiou: int
    best_piou: int
    best_epoch: int
    keep_epoch: int
    val_num: int
    best_miou_m: int
    best_miou_b: int
    best_FBiou_m: int
    start_time: float
    train_average_meter: Dict[str, AverageMeter]
    val_average_meter: Dict[str, AverageMeter]
    class_intersection_meter: List
    class_union_meter: List
    class_intersection_meter_m: List
    class_union_meter_m: List
    class_intersection_meter_b: List
    class_union_meter_b: List
    class_target_meter_b: List
    class_intersection_meter: List
    class_intersection_meter: List
    class_intersection_meter: List
    test_num: int
    base_label_num: int
    novel_label_num: int

    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.args = args
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dinov2 = True if args.layers == 'dinov2' else False
        self.vis = False
        self.learning_rate = self.args.base_lr
        self.dataset = args.data_set
        self.BAM_weight = args.BAM_weight if args.enable_BAM else 0.0
        self.enable_BAM = args.enable_BAM

        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        elif self.dataset == 'FSS1000':
            self.base_classes = 520
        
        assert self.layers in [50, 101, 152, 'dinov2']
    
        self.PSPNet_ = PSPNet(args)

        if args.enable_BAM:
            backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
            weight_path = f'initmodel/PSPNet/{args.data_set}/split{args.split}/{backbone_str}/best.pth'
            new_param: OrderedDict = torch.load(weight_path, map_location=self.device)['state_dict']
            try:
                self.PSPNet_.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                self.PSPNet_.load_state_dict(new_param)
        
        
        # Base Learner
        self.learner_base = nn.Sequential(self.PSPNet_.ppm, self.PSPNet_.cls)

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        elif self.dinov2:
            fea_dim = 1024 + 1024
        else:
            fea_dim = 1024 + 512

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        
        mask_add_num = 1

        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*3 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        
        self.block_num = self.args.decoder_block_num
        self.cross_num = min(self.args.cross_block_num, self.args.decoder_block_num)
        self.S_block_res2_meta = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1)) for _ in range(self.block_num)])
        self.Q_block_res2_meta = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1)) for _ in range(self.block_num)])
        
        self.Q_cls_head = nn.ModuleList([nn.Sequential(            
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1),
            ) for _ in range(self.block_num-1)])

        self.S_cls_head = nn.ModuleList([nn.Sequential(            
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1),
            ) for _ in range(self.block_num-1)])

        self.Q_cls_meta = nn.Sequential(
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))
        self.S_cls_meta = nn.Sequential(
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))
        
        self.Q_block: nn.ModuleList[SA] = nn.ModuleList([
            SA(in_channels=256) for _ in range(self.block_num)
        ])

        self.S_block: nn.ModuleList[SA] = nn.ModuleList([
            SA(in_channels=256) for _ in range(self.block_num)
        ])

        self.CrossAttn: nn.ModuleList[CrossAttn] = nn.ModuleList([
            # ChannelCrossAttn(dim=reduce_dim*2) for _ in range(self.cross_num)
            CrossAttn(in_channels=256) for _ in range(self.cross_num)
        ])

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()


        # CAM
        self.cam_dict: Dict[str, CAMWrapper] = {
            'BagCAM': BagCAM,
            'GradCAM': GradCAM,
            'GradCAMPlusPlus': GradCAMPlusPlus, 
            'XGradCAM': XGradCAM, 
            'LayerCAM': LayerCAM, 
            'EigenCAM': EigenCAM, 
            'EigenGradCAM': EigenGradCAM, 
            'HiResCAM': HiResCAM
        }
        self.cam_net = self.cam_dict[args.method](self.PSPNet_)
        
        if self.dinov2:
            self.cam_mask_init = nn.Embedding(num_embeddings=4, embedding_dim=1024)
        else:
            self.cam_mask_init = nn.Embedding(num_embeddings=4, embedding_dim=2048)
        self.cam_mask_init.weight.data = self.cam_mask_init.weight.data / 100


        dm = FSSDatasetModule(args)
        steps_per_epoch = len(dm.train_dataloader())
        self.max_steps = steps_per_epoch * self.args.epochs
        
        self.example_input_array = next(iter(dm.train_dataloader()))
        self.freeze_modules()

        print(self.PSPNet_)

        # self.check_data(dm)

    def check_data(self, dm: FSSDatasetModule):
        dm.train_dataloader().dataset.check()
        dm.val_dataloader().dataset.check()
        dm.test_dataloader().dataset.check()
        raise Exception('Stop here.')

    def train_mode(self):
        self.train()
        self.PSPNet_.eval()
    
    def training_step(self, batch, batch_idx):
        '''
        batch: Dict
            x,
            y_m,
            y_b,
            s_x,
            s_y,
            cat_idx,
        '''

        x = batch['x']
        y_m: torch.Tensor = batch['y_m']
        s_x = batch['s_x']
        s_y = batch['s_y']
        cat_idx = batch['cat_idx']

        n = x.size(0) # batch_size

        output, main_loss, aux_loss1, aux_loss2, mask_ratio_list = self.forward(s_x=s_x, s_y=s_y, x=x, y_m=y_m, cat_idx=cat_idx)
        loss = self.args.BAM_weight*main_loss + self.args.meta_weight*aux_loss1 + self.args.middle_weight*aux_loss2

        intersection, union, target = intersectionAndUnionGPU(output, y_m, self.args.classes, self.args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()        

        self.train_average_meter['intersection_meter'].update(intersection)
        self.train_average_meter['union_meter'].update(union)
        self.train_average_meter['target_meter'].update(target)

        accuracy = sum(self.train_average_meter['intersection_meter'].val)*100 / (sum(self.train_average_meter['target_meter'].val) + 1e-10)  # allAcc
        
        self.train_average_meter['main_loss_meter'].update(main_loss.item(), n)
        self.train_average_meter['aux_loss_meter1'].update(aux_loss1.item(), n)
        self.train_average_meter['aux_loss_meter2'].update(aux_loss2.item(), n)
        self.train_average_meter['loss_meter'].update(loss.item(), n)

        self.train_average_meter['batch_time'].update(time.time() - self.trn_end - self.val_time)
        self.trn_end = time.time()

        for cross_num in range(mask_ratio_list.shape[0]):
            self.train_average_meter[f'mask_ratio_c{cross_num}_meter'].update(mask_ratio_list[cross_num].mean().item(), mask_ratio_list.shape[1])


        max_step = self.args.epochs * len(self.trainer.train_dataloader)
        remain_iter = max_step - self.trainer.global_step
        remain_time = remain_iter * self.train_average_meter['batch_time'].avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)

        if (batch_idx+1) % self.args.print_freq == 0:
            self.print(f'Epoch: [{self.current_epoch+1}/{self.args.epochs}][{batch_idx+1}/{len(self.trainer.train_dataloader)}] '
                        f'Batch {self.train_average_meter["batch_time"].val:.3f} ({self.train_average_meter["batch_time"].avg:.3f}) '
                        f'Remain {int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d} '
                        f'MainLoss {self.train_average_meter["main_loss_meter"].val:.4f} '
                        f'AuxLoss1 {self.train_average_meter["aux_loss_meter1"].val:.4f} '                        
                        f'AuxLoss2 {self.train_average_meter["aux_loss_meter2"].val:.4f} '                        
                        f'Loss {self.train_average_meter["loss_meter"].val:.4f} '
                        f'Accuracy {accuracy:.4f}.')
            
            trn_logdict = {
                'trn/Loss': self.train_average_meter['loss_meter'].val,
                'trn/MainLoss': self.train_average_meter['main_loss_meter'].val,
                'trn/AuxLoss1': self.train_average_meter['aux_loss_meter1'].val,
                'trn/AuxLoss2': self.train_average_meter['aux_loss_meter2'].val,
                }

            for cross_num in range(self.cross_num):
                for shot in range(self.shot):
                    trn_logdict[f'trn/mask_ratio_c{cross_num}'] = self.train_average_meter[f'mask_ratio_c{cross_num}_meter'].avg


            for k, v in trn_logdict.items():
                self.log(k, v, on_step=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        setup_seed(self.args.manual_seed+self.current_epoch, self.args.seed_deterministic)

    def on_train_epoch_end(self):
        iou_class = self.train_average_meter['intersection_meter'].sum / (self.train_average_meter['union_meter'].sum + 1e-10)
        accuracy_class = self.train_average_meter['intersection_meter'].sum / (self.train_average_meter['target_meter'].sum + 1e-10)
        FBIoU = np.mean(iou_class) * 100
        mIoU = np.mean(self.train_average_meter['intersection_meter'].val / self.train_average_meter['union_meter'].val)
        mAcc = np.mean(accuracy_class) * 100
        allAcc = sum(self.train_average_meter['intersection_meter'].sum)*100 / (sum(self.train_average_meter['target_meter'].sum) + 1e-10)

        self.print(f'Train result at epoch [{self.current_epoch}/{self.args.epochs}]: FBIoU/mAcc/allAcc {FBIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}.')
        for i in range(self.args.classes):
            self.print(f'Class_{i} Result: iou/accuracy {iou_class[i]*100:.4f}/{accuracy_class[i]*100:.4f}.')
        
        self.log('trn/FBIoU', FBIoU, on_epoch=True, logger=True, sync_dist=True)
        self.log('trn/mIoU', mIoU, on_epoch=True, logger=True, sync_dist=True)
        

    def validation_step(self, batch:Dict, batch_idx):
        '''
        'x': x,
        'y_m': y_m,
        'label_b': label_b,
        's_x': s_x,
        's_y': s_y,
        'subcls_list': subcls_list,
        'raw_label': raw_label,
        'raw_label_b': raw_label_b
        '''

        x: torch.Tensor = batch['x']          # [b, c, h, w]
        y_m: torch.Tensor = batch['y_m']      # [b, h, w]
        y_b: torch.Tensor = batch['y_b']      # [b, h, w]        
        s_x: torch.Tensor = batch['s_x']      # [b, s, c, h, w]
        s_y: torch.Tensor = batch['s_y']      # [b, s, h, w]
        cat_idxs = batch['cat_idx']
        ori_labels: torch.Tensor = batch['raw_label']
        ori_label_bs: torch.Tensor = batch['raw_label_b']
        ori_label_shapes: torch.Tensor = batch['raw_label_shape']
        ori_label_b_shapes: torch.Tensor = batch['raw_label_b_shape']
        q_name = batch.get('q_name', None)
        s_name = batch.get('s_name', None)

        criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignore_label)

        self.val_average_meter['data_time'].update(time.time() - self.val_end)
        start_time = time.time()

        output_list, query_meta_out_list, base_out_list = [], [], []

        output, query_meta_out, base_out, sim_measurement, mask_ratio_list = self.forward(s_x=s_x, s_y=s_y, x=x, y_m=y_m, cat_idx=cat_idxs, q_name=q_name, s_name=s_name, TTA_vis=True)
        self.val_average_meter['sim_meter'].update(sim_measurement)

        output_list.append(output)
        query_meta_out_list.append(query_meta_out)
        base_out_list.append(base_out)

        if self.args.TTA:
            # 0.8 TTA
            x = F.interpolate(x, scale_factor=0.8, mode='bilinear', align_corners=True)
            s_x = torch.cat([F.interpolate(s_x[:,i,:,:,:], scale_factor=0.8, mode='bilinear', align_corners=True).unsqueeze(1) for i in range(self.shot)], dim=1)
            s_y = torch.cat([F.interpolate(s_y[:,i,:,:].unsqueeze(1).float(), scale_factor=0.8, mode='nearest').long() for i in range(self.shot)], dim=1)
            y_m_1= F.interpolate(y_m.unsqueeze(1).float(), scale_factor=0.8, mode='nearest').squeeze(1).long()
            output, query_meta_out, base_out, _, _ = self.forward(s_x=s_x, s_y=s_y, x=x, y_m=y_m_1, cat_idx=cat_idxs, q_name=q_name, s_name=s_name, TTA_vis=False)

            output = F.interpolate(output, size=(self.args.train_h, self.args.train_w), mode='bilinear', align_corners=True)
            query_meta_out = F.interpolate(query_meta_out, size=(self.args.train_h, self.args.train_w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(self.args.train_h, self.args.train_w), mode='bilinear', align_corners=True)

            output_list.append(output)
            query_meta_out_list.append(query_meta_out)
            base_out_list.append(base_out)

            # 1.25 TTA
            x = F.interpolate(x, scale_factor=1.25, mode='bilinear')
            s_x = torch.cat([F.interpolate(s_x[:,i,:,:,:], scale_factor=1.25, mode='bilinear', align_corners=True).unsqueeze(1) for i in range(self.shot)], dim=1)
            s_y = torch.cat([F.interpolate(s_y[:,i,:,:].unsqueeze(1).float(), scale_factor=1.25, mode='nearest').long() for i in range(self.shot)], dim=1)
            y_m_2 = F.interpolate(y_m.unsqueeze(1).float(), scale_factor=1.25, mode='nearest').squeeze(1).long()
            output, query_meta_out, base_out, _, _ = self.forward(s_x=s_x, s_y=s_y, x=x, y_m=y_m_2, cat_idx=cat_idxs, q_name=q_name, s_name=s_name, TTA_vis=False)

            output = F.interpolate(output, size=(self.args.train_h, self.args.train_w), mode='bilinear', align_corners=True)
            query_meta_out = F.interpolate(query_meta_out, size=(self.args.train_h, self.args.train_w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(self.args.train_h, self.args.train_w), mode='bilinear', align_corners=True)

            output_list.append(output)
            query_meta_out_list.append(query_meta_out)
            base_out_list.append(base_out)
        
        outputs = torch.stack(output_list, dim=0).mean(dim=0)
        query_meta_outs = torch.stack(query_meta_out_list, dim=0).mean(dim=0)
        base_outs = torch.stack(base_out_list, dim=0).mean(dim=0)


        self.val_average_meter['model_time'].update(time.time() - start_time)
        for output, query_meta_out, base_out, ori_label, ori_label_b, ori_label_shape, ori_label_b_shape, cat_idx in zip(outputs, query_meta_outs, base_outs, ori_labels, ori_label_bs, ori_label_shapes, ori_label_b_shapes, cat_idxs):

            ori_label = F.interpolate(repeat(ori_label, 'h w -> b c h w', b=1, c=1).float(), size=(ori_label_shape[0], ori_label_shape[1]), mode='nearest').squeeze(1).long()
            ori_label_b = F.interpolate(repeat(ori_label_b, 'h w -> b c h w', b=1, c=1).float(), size=(ori_label_b_shape[0], ori_label_b_shape[1]), mode='nearest').squeeze(1).long()

            target = ori_label.clone().long()
            target_b = ori_label_b.clone().long()
            
            if self.args.ori_resize:  # 真值转化为方形
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside, device=self.device)*255
                backmask_b = torch.ones(ori_label.size(0), longerside, longerside, device=self.device)*255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                backmask_b[0, :ori_label.size(1), :ori_label.size(2)] = ori_label_b
                target = backmask.clone().long()
                target_b = backmask_b.clone().long()

            output = F.interpolate(output.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
            query_meta_out = F.interpolate(query_meta_out.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
        
            loss = criterion(output, target)

            output = output.max(1)[1]
            query_meta_out = query_meta_out.max(1)[1]
            base_out = base_out.max(1)[1]

            cat_idx = cat_idx.cpu().numpy()

            intersection, union, new_target = intersectionAndUnionGPU(output, target, self.args.classes, self.args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()

            self.val_average_meter['intersection_meter'].update(intersection)
            self.val_average_meter['union_meter'].update(union)
            self.val_average_meter['target_meter'].update(new_target)

            self.class_intersection_meter[cat_idx] += intersection[1]
            self.class_union_meter[cat_idx] += union[1]
            
            intersection, union, new_target = intersectionAndUnionGPU(query_meta_out, target, self.args.classes, self.args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            self.val_average_meter['intersection_meter_m'].update(intersection)
            self.val_average_meter['union_meter_m'].update(union)
            self.val_average_meter['target_meter_m'].update(new_target)
            self.class_intersection_meter_m[cat_idx] += intersection[1]
            self.class_union_meter_m[cat_idx] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(base_out, target_b, self.base_label_num, self.args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            for idx in range(1,len(intersection)):
                self.class_intersection_meter_b[idx-1] += intersection[idx]
                self.class_union_meter_b[idx-1] += union[idx]
                self.class_target_meter_b[idx-1] += new_target[idx]

            accuracy = sum(self.val_average_meter['intersection_meter'].val)*100 / (sum(self.val_average_meter['target_meter'].val) + 1e-10)
            self.val_average_meter['loss_meter'].update(loss.item(), x.size(0))
        
        self.val_average_meter['batch_time'].update(time.time() - self.val_end)
        self.val_end = time.time()
        if ((batch_idx + 1) % round((self.test_num/100)) == 0):
            self.print(f'Test: [{(batch_idx + 1)*self.args.batch_size}/{self.test_num}] '
                        f'Data {self.val_average_meter["data_time"].val:.3f} ({self.val_average_meter["data_time"].avg:.3f}) '
                        f'Batch {self.val_average_meter["batch_time"].val:.3f} ({self.val_average_meter["batch_time"].avg:.3f}) '
                        f'Loss {self.val_average_meter["loss_meter"].val:.4f} ({self.val_average_meter["loss_meter"].avg:.4f}) '
                        f'Sim {self.val_average_meter["sim_meter"].val:.4f} ({self.val_average_meter["sim_meter"].avg:.4f}) '
                        f'Accuracy {accuracy:.4f}.'
                        )

        return self.val_average_meter['loss_meter'].val

    def on_validation_epoch_end(self):
        self.print('<<<<<<<<<<<<<<<<< Start Evaluation <<<<<<<<<<<<<<<<<')

        val_time = time.time() - self.val_start

        iou_class = self.val_average_meter['intersection_meter'].sum / (self.val_average_meter['union_meter'].sum + 1e-10)
        iou_class_m = self.val_average_meter['intersection_meter_m'].sum / (self.val_average_meter['union_meter_m'].sum + 1e-10)
        FBIoU = np.mean(iou_class) * 100
        FBIoU_m = np.mean(iou_class_m) * 100
        
        class_iou_class = []
        class_iou_class_m = []
        class_iou_class_b = []
        class_miou = 0
        class_miou_m = 0
        class_miou_b = 0
        for i in range(len(self.class_intersection_meter)):
            class_iou = self.class_intersection_meter[i]/(self.class_union_meter[i]+ 1e-10)
            class_iou_class.append(class_iou)
            class_miou += class_iou
            class_iou = self.class_intersection_meter_m[i]/(self.class_union_meter_m[i]+ 1e-10)
            class_iou_class_m.append(class_iou)
            class_miou_m += class_iou
        for i in range(len(self.class_intersection_meter_b)):
            class_iou = self.class_intersection_meter_b[i]/(self.class_union_meter_b[i]+ 1e-10)
            class_iou_class_b.append(class_iou)
            class_miou_b += class_iou

        target_b = np.array(self.class_target_meter_b)

        mIoU = class_miou*100.0 / len(self.class_intersection_meter)
        mIoU_m = class_miou_m*100.0 / len(self.class_intersection_meter)
        mIoU_b = class_miou_b*100.0 / (len(self.class_intersection_meter_b) - len(target_b[target_b==0]))  # filter the results with GT mIoU=0

        if self.enable_BAM:
            self.print(f'meanIoU---Val result: mIoU_f {mIoU:.4f}.')     # final
        self.print(f'meanIoU---Val result: mIoU_m {mIoU_m:.4f}.')   # meta
        if self.enable_BAM:
            self.print(f'meanIoU---Val result: mIoU_b {mIoU_b:.4f}.')   # base
        self.print(f'meanBCE---Val result: sim BCE {self.val_average_meter["sim_meter"].avg:.4f}.')   # similar map

        self.print('<<<<<<< Novel Results <<<<<<<')
        for i in range(self.novel_label_num):
            if self.enable_BAM:
                self.print(f'Class_{i+1} Result: iou_f {class_iou_class[i]*100:.4f}.')         
            self.print(f'Class_{i+1} Result: iou_m {class_iou_class_m[i]*100:.4f}.')

        if self.enable_BAM:
            self.print('<<<<<<< Base Results <<<<<<<')
            for i in range(self.base_label_num-1):
                if self.class_target_meter_b[i] == 0:
                    self.print(f'Class_{i+1+self.novel_label_num} Result: iou_b None.')
                else:
                    self.print(f'Class_{i+1+self.novel_label_num} Result: iou_b {class_iou_class_b[i]*100:.4f}.')

        if self.enable_BAM:
            self.print(f'FBIoU---Val result: FBIoU_f {FBIoU:.4f}.')
        self.print(f'FBIoU---Val result: FBIoU_m {FBIoU_m:.4f}.')
        for i in range(self.args.classes):
            if self.enable_BAM:
                self.print(f'Class_{i} Result: iou_f {iou_class[i]*100:.4f}.')
            self.print(f'Class_{i} Result: iou_m {iou_class_m[i]*100:.4f}.')
        self.print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        self.print(f'total time: {val_time:.4f}, avg inference time: {self.val_average_meter["model_time"].avg:.4f}, count: {self.test_num}')

        val_logdict = {
            'val/Loss': self.val_average_meter['loss_meter'].avg,
            'val/FBIoU': FBIoU,
            'val/FBIoU_m': FBIoU_m,
            'val/mIoU': mIoU,
            'val/mIoU_m': mIoU_m,
            'val/sim': self.val_average_meter['sim_meter'].avg
            }

        if mIoU > self.best_miou:
            self.best_miou, self.best_miou_m, self.best_miou_b, self.best_epoch = mIoU, mIoU_m, mIoU_b, self.current_epoch
            self.best_FBiou, self.best_FBiou_m, self.best_piou = FBIoU, FBIoU_m, iou_class[1]

        val_logdict['val/best_mIoU'] = self.best_miou
        for k, v in val_logdict.items():
            self.log(k, v, on_epoch=True, logger=True, sync_dist=True)
        
    def on_train_end(self):
        total_time = time.time() - self.start_time
        t_m, t_s = divmod(total_time, 60)
        t_h, t_m = divmod(t_m, 60)
        total_time = f'{int(t_h):02d}h {int(t_m):02d}m {int(t_s):02d}s'
        self.print(f'\nEpoch: {self.current_epoch}/{self.args.epochs} \t Total running time: {total_time}')
        self.print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        self.print(f'{self.args.arch}\t Group:{self.args.split} \t Best_step:{self.best_epoch}')
        self.print(f'mIoU:{self.best_miou:.4f} \t mIoU_m:{self.best_miou_m:.4f} \t mIoU_b:{self.best_miou_b:.4f}')
        self.print(f'FBIoU:{self.best_FBiou:.4f} \t FBIoU_m:{self.best_FBiou_m:.4f} \t pIoU:{self.best_piou:.4f}')
        self.print('>'*80)
        self.print (f'{datetime.now()}')

    def test_step(self, batch, batch_idx):
        self.vis = self.args.vis
        torch.set_grad_enabled(True)
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):

        optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'ScheduleFreeAdamW': ScheduleFreeAdamW
        }
        optimizer_cls = optimizer_dict[self.args.optimizer]
        if self.args.shot > 1:
            optimizer_kwargs = dict(lr=self.learning_rate, weight_decay=self.args.weight_decay)
            if self.args.optimizer == 'SGD':
                optimizer_kwargs['momentum'] = self.args.momentum
            optimizer = optimizer_cls(
                [     
                {'params': self.down_query.parameters()},
                {'params': self.down_supp.parameters()},
                {'params': self.init_merge.parameters()},
                {'params': self.Q_block.parameters()},
                {'params': self.S_block.parameters()},
                {'params': self.CrossAttn.parameters()},
                {'params': self.S_block_res2_meta.parameters()},
                {'params': self.Q_block_res2_meta.parameters()},
                {'params': self.S_cls_head.parameters()},
                {'params': self.Q_cls_head.parameters()},                   
                {'params': self.Q_cls_meta.parameters()},
                {'params': self.S_cls_meta.parameters()},
                {'params': self.gram_merge.parameters()},
                {'params': self.cls_merge.parameters()},
                {'params': self.kshot_rw.parameters()},
                # {'params': self.cam_mask_init.parameters()}
                ], **optimizer_kwargs)
        else:
            optimizer_kwargs = dict(lr=self.learning_rate, weight_decay=self.args.weight_decay)
            if self.args.optimizer == 'SGD':
                optimizer_kwargs['momentum'] = self.args.momentum
            optimizer = optimizer_cls(
                [
                {'params': self.down_query.parameters()},
                {'params': self.down_supp.parameters()},
                {'params': self.init_merge.parameters()},
                {'params': self.Q_block.parameters()},
                {'params': self.S_block.parameters()},
                {'params': self.CrossAttn.parameters()},
                {'params': self.S_block_res2_meta.parameters()},
                {'params': self.Q_block_res2_meta.parameters()},
                {'params': self.S_cls_head.parameters()},
                {'params': self.Q_cls_head.parameters()},  
                {'params': self.Q_cls_meta.parameters()},
                {'params': self.S_cls_meta.parameters()},
                {'params': self.gram_merge.parameters()},
                {'params': self.cls_merge.parameters()},        
                # {'params': self.cam_mask_init.parameters()}
                ], **optimizer_kwargs)

        # cosine+warmup
        # 定义 lambda 函数实现 Warmup + Cosine 衰减
        # def lr_lambda(current_step: int):
        #     total_steps = self.max_steps  # 总训练步数
        #     warmup_steps = int(0.1 * total_steps)  # 10% 预热步数
            
        #     # Warmup 阶段：线性增加学习率
        #     if current_step < warmup_steps:
        #         return float(current_step) / max(1, warmup_steps)  # 从 0 → base_lr
            
        #     # Cosine 衰减阶段
        #     decay_steps = total_steps - warmup_steps  # 衰减阶段总步数
        #     cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / decay_steps))
        #     return cosine_decay  # 从 base_lr → 0

        # # 创建 LambdaLR 调度器
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, 
        #     lr_lambda=lr_lambda
        # )

        lr_lambda = lambda step: (1 - float(step) / self.max_steps) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},}
    
    def get_progress_bar_dict(self):
        # to stop to show the version number in the progress bar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def freeze_modules(self):
        for param in self.learner_base.parameters():
            param.requires_grad = False
        for param in self.PSPNet_.layer0.parameters():
            param.requires_grad = False
        for param in self.PSPNet_.layer1.parameters():
            param.requires_grad = False
        for param in self.PSPNet_.layer2.parameters():
            param.requires_grad = False

    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x: torch.Tensor, s_x: torch.Tensor, s_y: torch.Tensor, y_m: torch.Tensor, TTA_vis=False, y_b=None, cat_idx=None, q_name=None, s_name=None):
        '''        
        cat_idx: the index of select class in sub_list (e.g. 0~15 in pascal fold3)
        '''

        B, C, H, W = x.size()
        # self.cam_mask_init = self.cam_mask_init.to(device=x.device)
        self.cam_net.device = self.device

        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.PSPNet_.layer0(x)

            if self.dinov2:
                query_feat_0 = torch.cat((self.PSPNet_.cls_token.expand(B, -1, -1), query_feat_0), dim=1)
                query_feat_0 = query_feat_0 + self.PSPNet_.interpolate_pos_encoding(query_feat_0, H, W)

            query_feat_1 = self.PSPNet_.layer1(query_feat_0)
            query_feat_2 = self.PSPNet_.layer2(query_feat_1)
            query_feat_3 = self.PSPNet_.layer3(query_feat_2)
            query_feat_4 = self.PSPNet_.layer4(query_feat_3)

            query_feat_list: List[torch.Tensor] = []
            if self.dinov2:
                for i in range(5):
                    query_low_feat = eval('query_feat_' + self.low_fea_id)[:, 1:, :]  # remove cls token
                    query_feat_list.append(BNC2BCHW(query_low_feat))
            else:
                query_feat_list = [query_feat_0, query_feat_1, query_feat_2, query_feat_3, query_feat_4]

        # Support Feature
        supp_imgs: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        supp_low_feat_list: List[torch.Tensor] = []
        supp_feat_list: List[torch.Tensor] = []
        supp_cls_token_list = []
        supp_cls_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            supp_imgs.append(s_x[:,i,:,:,:])

            with torch.no_grad():


                if not self.dinov2:

                    supp_feat_0 = self.PSPNet_.layer0(s_x[:,i,:,:,:])

                    supp_feat_1 = self.PSPNet_.layer1(supp_feat_0)
                    supp_feat_2 = self.PSPNet_.layer2(supp_feat_1)
                    supp_feat_3 = self.PSPNet_.layer3(supp_feat_2)

                    mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                    supp_feat_4 = self.PSPNet_.layer4(supp_feat_3*mask)

                    supp_cls = self.PSPNet_.avgpool(supp_feat_4)
                    supp_cls = supp_cls.flatten(1)
                    supp_cls = self.PSPNet_.fc(supp_cls)
                    supp_cls_list.append(supp_cls)
                    supp_feat_list.append([supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4])               

                else:

                    supp_feat_0 = self.PSPNet_.layer0(s_x[:,i,:,:,:])
                    supp_feat_0 = torch.cat((self.PSPNet_.cls_token.expand(B, -1, -1), supp_feat_0), dim=1)
                    supp_feat_0 = supp_feat_0 + self.PSPNet_.interpolate_pos_encoding(supp_feat_0, H, W)

                    supp_feat_1 = self.PSPNet_.layer1(supp_feat_0)
                    supp_feat_2 = self.PSPNet_.layer2(supp_feat_1)
                    supp_feat_3 = self.PSPNet_.layer3(supp_feat_2)
                    supp_feat_4 = self.PSPNet_.layer4(supp_feat_3)  # (B, N, 1024)

                    cls_token = supp_feat_4[:, 0]
                    supp_cls_token_list.append(cls_token)
                    patch_tokens = supp_feat_4[:, 1:]

                    linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
                    supp_cls = self.PSPNet_.fc(linear_input)  # (B, 1000)

                    supp_cls_list.append(supp_cls)

                    tmp = []
                    for i in range(5):
                        supp_low_feat = eval('supp_feat_' + self.low_fea_id)[:, 1:, :]  # remove cls token
                        tmp.append(BNC2BCHW(supp_low_feat))
                    supp_feat_list.append(tmp)                    

            if self.dinov2:
                # Convert [B, N, C] to [B, C, H, W] for dinov2
                supp_low_feat = eval('supp_feat_' + self.low_fea_id)[:, 1:, :]  # remove cls token
                supp_low_feat_list.append(BNC2BCHW(supp_low_feat))
            else:
                supp_low_feat_list.append(eval('supp_feat_' + self.low_fea_id))


        # K-Shot Reweighting
        if self.dinov2:
            query_low_feat = eval('query_feat_' + self.low_fea_id)[:, 1:, :]
            query_low_feat = BNC2BCHW(query_low_feat)
        else:
            query_low_feat = eval('query_feat_' + self.low_fea_id)

        que_gram = get_gram_matrix(query_low_feat) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in supp_low_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(bs,1,1,1).contiguous()) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight:torch.Tensor = self.kshot_rw(val1)       # channel shot->2->shot
            weight = weight.gather(1, idx2)                 # dim=1, index=idx2
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)    # [bs, 1, 1, 1]

        # Base and Meta

        # query_feat_0 = self.PSPNet_.layer0(x)
        # query_feat_1 = self.PSPNet_.layer1(query_feat_0)
        # query_feat_2 = self.PSPNet_.layer2(query_feat_1)
        # query_feat_3 = self.PSPNet_.layer3(query_feat_2)
        # query_feat_4 = self.PSPNet_.layer4(query_feat_3)
        # self.learner_base = nn.Sequential(self.PSPNet_.ppm, self.PSPNet_.cls)

        if self.dinov2:
            query_feat_4 = query_feat_4[:, 1:, :]
            query_feat_4 = BNC2BCHW(query_feat_4)

        base_out: torch.Tensor = self.learner_base(query_feat_4)              # [bs, 16, 15, 15]

        query_meta_out, supp_meta_out_list, new_y_m, new_mask_list, S_middle_out, Q_middle_out, sim_measurement, mask_ratio_list = self.meta_forward(
            query_img=x,
            supp_imgs=supp_imgs, 
            query_feat_list=query_feat_list,
            supp_feat_list=supp_feat_list,
            supp_cls_list=supp_cls_list,
            query_mask=y_m,
            supp_mask_list=mask_list,
            weight_soft=weight_soft,
            q_name=q_name,
            s_names=s_name,
            q_cls_token=query_feat_4[:,0,:] if self.dinov2 else None,
            s_cls_tokens=supp_cls_token_list if self.dinov2 else None,
            TTA_vis=TTA_vis
            )                                                   # [bs, 2, 30, 30]

        if self.enable_BAM:
            meta_out_soft = query_meta_out.softmax(1)
            base_out_soft = base_out.softmax(1)

            # Classifier Ensemble
            meta_map_bg = meta_out_soft[:,:1,:,:]                           # [bs, 1, 30, 30]
            meta_map_fg = meta_out_soft[:,1:,:,:]                           # [bs, 1, 30, 30]
            if self.training and self.cls_type == 'Base':
                c_id_array = torch.arange(self.base_classes+1, device=self.device)       # 16 for pascal
                base_map_list = []
                for b_id in range(bs):      # batch_size
                    c_id = cat_idx[b_id] + 1     # selected class in batch
                    c_mask = (c_id_array!=0)&(c_id_array!=c_id)     # background(0) and base selected class is False, else is True
                    base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
                base_map = torch.cat(base_map_list,0)
                # <alternative implementation>
                # gather_id = (cat_idx+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
                # fg_map = base_out_soft.gather(1,gather_id)
                # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map
            else:
                base_map = base_out_soft[:,1:,:,:].sum(1,True)

            est_map = est_val.expand_as(meta_map_fg)        # [bs, 1, 1, 1] to # [bs, 1, 30, 30]

            # nn.Conv2d(2, 1, k=1) initialize as [1,0]
            meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))  # [bs, 2, 30, 30] to [bs, 1, 30, 30]
            meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

            # nn.Conv2d(2, 1, k=1) initialize as [1,0]
            base_map = F.interpolate(base_map, size=(meta_map_bg.size(2),meta_map_bg.size(3)), mode='bilinear', align_corners=True)
            merge_map = torch.cat([meta_map_bg, base_map], 1)           # [bs, 2, 30, 30]
            merge_bg = self.cls_merge(merge_map)                        # [bs, 2, 30, 30] to [bs, 1, 30, 30]

            final_out = torch.cat([merge_bg, meta_map_fg], dim=1)
        
        else:
            final_out = query_meta_out

        # Output Part
        if self.zoom_factor != 1:
            query_meta_out = F.interpolate(query_meta_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            

            for i in range(self.shot):
                supp_meta_out_list[i] = F.interpolate(supp_meta_out_list[i], size=(h, w), mode='bilinear', align_corners=True)
                new_mask_list[i] = F.interpolate(new_mask_list[i].float(), size=(h, w), mode='bilinear', align_corners=True)
            

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(query_meta_out, y_m.long())

            S_middle_out_loss = torch.empty((self.block_num-1, self.shot), device=final_out.device)
            Q_middle_out_loss = torch.empty((self.block_num-1), device=final_out.device)
            for i in range(self.block_num-1):
                for j in range(self.shot):
                    s_middle = F.interpolate(S_middle_out[i, j].float(), size=(h, w), mode='bilinear', align_corners=True)
                    S_middle_out_loss[i, j] = self.criterion(s_middle, mask_list[j].squeeze(dim=1).long())

                q_middle = F.interpolate(Q_middle_out[i].float(), size=(h, w), mode='bilinear', align_corners=True)
                Q_middle_out_loss[i] = self.criterion(q_middle, y_m.long())

            # mask_list: shot list of [b, 1, h, w]
            supp_loss = sum([self.criterion(supp_meta_out, supp_mask.squeeze(dim=1).long()) for supp_meta_out, supp_mask in zip(supp_meta_out_list, mask_list)]) / self.shot

            aux_loss1 = (aux_loss1 + supp_loss) / 2
            aux_loss2 = (torch.mean(Q_middle_out_loss) + torch.mean(S_middle_out_loss)) / 2

            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2, mask_ratio_list
        else:
            if self.vis and TTA_vis:
                final_mask = final_out.max(1)[1]    # [b h w]
                final_mask = t2n(final_mask.long())
                b, h, w = final_mask.shape

                for i in range(b):
                    q_img = x[i]
                    q_vis_image = torch.empty_like(q_img, dtype=torch.uint8)
                    for c in range(q_img.shape[0]):
                        q_img_c = q_img[c]
                        q_img_c = (q_img_c - q_img_c.min()) / (q_img_c.max() - q_img_c.min()) * 255
                        q_vis_image[c] = q_img_c.to(torch.uint8)

                    q_vis_image = np.int64(q_vis_image.detach().cpu())
                    q_vis_image[q_vis_image > 255] = 255
                    q_vis_image[q_vis_image < 0] = 0
                    q_vis_image = np.uint8(q_vis_image.transpose(1,2,0))

                    q_vis_path = ospj(self.args.logpath, self.args.vis_name, q_name[i], 'q')
                    plt.imsave(ospj(q_vis_path)+f"_pred.png", generate_vis(final_mask[i], q_vis_image, rate=0.4, color='r'))
                    plt.imsave(ospj(q_vis_path)+f"_pred_mask.png", final_mask[i]*255)

                    q_img = x[i]
                    q_mask = y_m[i]
                    q_mask = t2n(q_mask.long())

                    plt.imsave(ospj(q_vis_path)+f"_original.png", q_vis_image)
                    plt.imsave(ospj(q_vis_path)+f"_fore_mask.png", q_mask*255)
                    plt.imsave(ospj(q_vis_path)+f"_fore.png", generate_vis(q_mask, q_vis_image, rate=0.4, color='r'))


                    q_mask = q_mask.reshape(h*w)
                    tmp_final_mask = final_mask[i].reshape(h*w)
                    mIoU = np.logical_and(q_mask, tmp_final_mask).sum()*100 / np.logical_or(q_mask, tmp_final_mask).sum()

                    rename_old_path = ospj(self.args.logpath, self.args.vis_name, q_name[i])
                    rename_new_path = ospj(self.args.logpath, self.args.vis_name, f"{q_name[i]}_{mIoU:.2f}")

                    if os.path.exists(rename_new_path):
                        import shutil
                        shutil.rmtree(rename_new_path, ignore_errors=True)
                    os.rename(rename_old_path, rename_new_path)

            return final_out, query_meta_out, base_out, sim_measurement, mask_ratio_list

    def meta_forward(self,
                     query_img: torch.Tensor, 
                     supp_imgs: List[torch.Tensor], 
                     query_feat_list: List[torch.Tensor], 
                     supp_feat_list: List[torch.Tensor], 
                     supp_cls_list: List[torch.Tensor], 
                     q_cls_token: Union[torch.Tensor, None],
                     s_cls_tokens: Union[List[torch.Tensor], None],
                     query_mask: torch.Tensor,
                     supp_mask_list: List[torch.Tensor],
                     weight_soft: torch.Tensor,
                     s_names: Union[None, List[str]],
                     q_name: Union[None, List[List[str]]],
                     TTA_vis: bool) -> torch.Tensor:
        '''
        :params: query_img, torch.Size([bz, 3, 473, 473])
        :params: supp_imgs, shot length of torch.Size([bz, 3, 473, 473])
        :params: query_feat_list, 5 length of torch.Size([bz, 64, 119, 119]), torch.Size([bz, 256, 119, 119]), torch.Size([bz, 512, 60, 60]), torch.Size([bz, 1024, 30, 30]), torch.Size([bz, 2048, 15, 15])
        :params: supp_feat_list: [shot, 5] length of torch.Size([bz, 64, 119, 119]), torch.Size([bz, 256, 119, 119]), torch.Size([bz, 512, 60, 60]), torch.Size([bz, 1024, 30, 30]), torch.Size([bz, 2048, 15, 15])
        :params: supp_cls_list: shot length of torch.Size([bz, 1000])
        :params: q_cls_token: Union[torch.Tensor, None], torch.Size([bz, 1024]) when dinov2 else None
        :params: s_cls_tokens: Union[List[torch.Tensor], None], torch.Size([bz, 1024]) * shot when dinov2 else None
        :params: query_mask, torch.Size([bz, 473, 473])
        :params: supp_mask_list, shot length of torch.Size([bz, 1, 473, 473])
        :params: weight_soft, torch.Size([bz, shot, 1, 1])
        :params: q_name, None when training, [str] length=1 when testing
        :params: s_name, None when training, [[str]*shot] when testing        
        :params: TTA_vis, dont vis during TTA

        :return: query_meta_out, [bz, 2, 30, 30]
        :return: supp_meta_out, shot length of [bz, 2, 30, 30]
        :return: rectified_query_mask, [bz, 1, 30, 30]
        :return: rectified_supp_masks, shot length of [bz, 1, 30, 30]
        :return: S_middle_out, [block_num-1, s, b, 256, 30, 30]
        :return: Q_middle_out, [block_num-1, b, 256, 30, 30]
        '''

        query_mask[query_mask == 255] = 0

        query_feat_2 = query_feat_list[2]
        query_feat_3 = query_feat_list[3]
        cam_query_feat = query_feat_list[4]

        query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)     # [bsz, 1024+512, 30, 30]
        query_feat = self.down_query(query_feat)                    # [bsz, 256, 30, 30]

        bsz, C, H, W = query_feat.shape
        _, _, H_mask, W_mask = query_img.shape        

        # CAM based Prior Similarity Mask
        corr_query_mask = torch.empty((bsz, self.shot, H, W), device=query_img.device)               # [bs, k, 30, 30]
        supp_pro_list = torch.empty((self.shot, bsz, C, 1), device=query_img.device)                    # [bs, 256, k, 1]
        supp_pcam_pmask_list = torch.empty((self.shot, bsz, C, 1), device=query_img.device)             # [bs, 256, k, 1]
        supp_pcam_nmask_list = torch.empty((self.shot, bsz, C, 1), device=query_img.device)             # [bs, 256, k, 1]
        supp_ncam_pmask_list = torch.empty((self.shot, bsz, C, 1), device=query_img.device)             # [bs, 256, k, 1]
        supp_ncam_nmask_list = torch.empty((self.shot, bsz, C, 1), device=query_img.device)             # [bs, 256, k, 1]

        corr_supp_mask_list = torch.empty((self.shot, bsz, 1, H, W), device=query_img.device)           # [k, bsz, 1, 30, 30]
        supp_pseudo_masks_list = torch.empty((self.shot, bsz, 1, H_mask, W_mask), device=query_img.device)        # [k, bsz, 1, 473, 473]
        decode_supp_feat = torch.empty((self.shot, bsz, C, H, W), device=query_img.device)              # [k, bsz, 256, 30, 30]

        mask_ratio_list = torch.empty((self.cross_num, self.shot, bsz), device=query_img.device)

        # k-shot
        for i, (tmp_supp_img, tmp_supp_feat, supp_mask, tmp_supp_cls) in enumerate(zip(supp_imgs, supp_feat_list, supp_mask_list, supp_cls_list)):

            batch = {
                'query_img': query_img,
                'support_imgs': tmp_supp_img,
                'query_mask': query_mask,                 # only for vis
                'support_masks': supp_mask,
            }

            if self.vis and TTA_vis:
                batch['q_name'] = q_name
                batch['s_name'] = s_names[i]

            with torch.enable_grad():
                # _ _ torch.Size([b, 1, h, w]) torch.Size([b, 1, h, w]) torch.Size([b, 1, h, w])
                _, _, support_masks, support_pseudo_masks, query_pseudo_masks = Generator_CAM(
                    self.args,
                    cam_net = self.cam_net,
                    batch=batch,
                    vis=(self.vis and TTA_vis),
                    img_fore_cls=torch.argmax(tmp_supp_cls, 1),
                    vis_path=os.path.join(self.args.logpath, self.args.vis_name),
                    threshold=self.args.threshold,
                    )

            supp_pseudo_masks_list[i] = support_pseudo_masks

            # torch.Size([b, 1, h, w])
            pcam_pmask = torch.logical_and(support_masks, support_pseudo_masks)
            ncam_pmask = torch.logical_and(support_masks,torch.logical_not(support_pseudo_masks))
            pcam_nmask = torch.logical_and(torch.logical_not(support_masks),support_pseudo_masks)
            ncam_nmask = torch.logical_and(torch.logical_not(support_masks), torch.logical_not(support_pseudo_masks))


            supp_feat_2, supp_feat_3 = tmp_supp_feat[2], tmp_supp_feat[3]
            cam_supp_feat = tmp_supp_feat[4]

            supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)       # [k, bsz, 256, 30, 30]
            decode_supp_feat[i] = supp_feat             # [k, bsz, 256, 30, 30]

            tmp_supp_mask = F.interpolate(supp_mask, size=(supp_feat.size(2),supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_pro = Weighted_GAP(supp_feat, tmp_supp_mask).detach()
            supp_pro_list[i] = supp_pro

            tmp_pcam_pmask = F.interpolate(pcam_pmask.float(), size=(supp_feat.size(2),supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_pcam_pmask_pro = Weighted_GAP(supp_feat, tmp_pcam_pmask)
            supp_pcam_pmask_list[i] = supp_pcam_pmask_pro

            tmp_pcam_pmask = F.interpolate(pcam_pmask.float(), size=(cam_supp_feat.size(2), cam_supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_pcam_pmask4_pro = Weighted_GAP(cam_supp_feat, tmp_pcam_pmask)
            
            tmp_ncam_pmask = F.interpolate(ncam_pmask.float(), size=(supp_feat.size(2),supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_ncam_pmask_pro = Weighted_GAP(supp_feat, tmp_ncam_pmask)
            supp_ncam_pmask_list[i] = supp_ncam_pmask_pro
     
            tmp_ncam_pmask = F.interpolate(ncam_pmask.float(), size=(cam_supp_feat.size(2), cam_supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_ncam_pmask4_pro = Weighted_GAP(cam_supp_feat, tmp_ncam_pmask)
    
            tmp_pcam_nmask = F.interpolate(pcam_nmask.float(), size=(supp_feat.size(2),supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_pcam_nmask_pro = Weighted_GAP(supp_feat, tmp_pcam_nmask)
            supp_pcam_nmask_list[i] = supp_pcam_nmask_pro

            tmp_pcam_nmask = F.interpolate(pcam_nmask.float(), size=(cam_supp_feat.size(2), cam_supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_pcam_nmask4_pro = Weighted_GAP(cam_supp_feat, tmp_pcam_nmask)

            tmp_ncam_nmask = F.interpolate(ncam_nmask.float(), size=(supp_feat.size(2),supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_ncam_nmask_pro = Weighted_GAP(supp_feat, tmp_ncam_nmask)
            supp_ncam_nmask_list[i] = supp_ncam_nmask_pro

            tmp_ncam_nmask = F.interpolate(ncam_nmask.float(), size=(cam_supp_feat.size(2), cam_supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_ncam_nmask4_pro = Weighted_GAP(cam_supp_feat, tmp_ncam_nmask)
        
            # query sim map
            tmp_query_pseudo_masks: torch.Tensor = F.interpolate(query_pseudo_masks.float(), size=(cam_query_feat.size(-2), cam_query_feat.size(-1)))

            q_forecam = cam_query_feat * tmp_query_pseudo_masks
            q_backcam = cam_query_feat * (1 - tmp_query_pseudo_masks)
            
            tmp_query_pseudo_masks = tmp_query_pseudo_masks.squeeze(1)      # [b,h,w]

            embedding = self.cam_mask_init(torch.tensor([0, 1, 2, 3], device=self.device).long())

            supp_pcam_pmask4_pro = rearrange(supp_pcam_pmask4_pro, "b c d -> b (c d)").detach()
            supp_pcam_pmask4_pro = supp_pcam_pmask4_pro + repeat(embedding[0], 'c -> n c', n=1)
            supp_pcam_pmask4_pro = supp_pcam_pmask4_pro / torch.norm(supp_pcam_pmask4_pro, p=2, dim=1, keepdim=True)

            supp_pcam_nmask4_pro = rearrange(supp_pcam_nmask4_pro, "b c d -> b (c d)").detach()
            supp_pcam_nmask4_pro = supp_pcam_nmask4_pro + repeat(embedding[1], 'c -> n c', n=1)
            supp_pcam_nmask4_pro = supp_pcam_nmask4_pro / torch.norm(supp_pcam_nmask4_pro, p=2, dim=1, keepdim=True)

            supp_ncam_pmask4_pro = rearrange(supp_ncam_pmask4_pro, "b c d -> b (c d)").detach()
            supp_ncam_pmask4_pro = supp_ncam_pmask4_pro + repeat(embedding[2], 'c -> n c', n=1)
            supp_ncam_pmask4_pro = supp_ncam_pmask4_pro / torch.norm(supp_ncam_pmask4_pro, p=2, dim=1, keepdim=True)

            supp_ncam_nmask4_pro = rearrange(supp_ncam_nmask4_pro, "b c d -> b (c d)").detach()
            supp_ncam_nmask4_pro = supp_ncam_nmask4_pro + repeat(embedding[3], 'c -> n c', n=1)
            supp_ncam_nmask4_pro = supp_ncam_nmask4_pro / torch.norm(supp_ncam_nmask4_pro, p=2, dim=1, keepdim=True)

            q_pcam_pmask = self.CossimCompute(q_forecam, supp_pcam_pmask4_pro)    # [b, h, w]
            q_pcam_nmask = self.CossimCompute(q_forecam, supp_pcam_nmask4_pro)    # [b, h, w]
            q_ncam_pmask = self.CossimCompute(q_backcam, supp_ncam_pmask4_pro)    # [b, h, w]
            q_ncam_nmask = self.CossimCompute(q_backcam, supp_ncam_nmask4_pro)    # [b, h, w]


            q_cam_mask = torch_minmaxnormalize(q_pcam_pmask, tmp_query_pseudo_masks) + torch_minmaxnormalize(q_ncam_pmask, 1 - tmp_query_pseudo_masks)
            
            if self.dinov2:
                # supp_cls_token: [B, C], query_feat_4: [B, C, H, W]
                # 计算supp_cls_token和query各patch之间的余弦相似度
                supp_cls_token = s_cls_tokens[i]  # [B, C]
                query_patch_tokens = rearrange(query_feat_list[4], 'b c h w -> b (h w) c')  # [B, N, C]

                # 归一化
                supp_cls_token_norm = supp_cls_token / (supp_cls_token.norm(dim=1, keepdim=True) + 1e-6)  # [B, 1024]
                query_patch_tokens_norm = query_patch_tokens / (query_patch_tokens.norm(dim=2, keepdim=True) + 1e-6)  # [B, N, 1024]

                # 计算余弦相似度
                cos_sim = torch.einsum('bd,bnd->bn', supp_cls_token_norm, query_patch_tokens_norm)  # [B, N]

                # 将patch相似度reshape为空间分布
                patch_h = int(H * W / cos_sim.shape[1]) if cos_sim.shape[1] != H * W else H
                patch_w = cos_sim.shape[1] // patch_h
                cls_patch_sim_map = cos_sim.view(-1, patch_h, patch_w)  # [B, H, W]
                corr_query = cls_patch_sim_map

            corr_query = repeat(q_cam_mask, 'b h w -> b c h w', c=1)


            # add sim map
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask[:, i] = rearrange(corr_query, 'b c h w -> (b c) h w')


            # supp sim map
            tmp_supp_pseudo_masks: torch.Tensor = F.interpolate(support_pseudo_masks.float(), size=(cam_supp_feat.size(-2), cam_supp_feat.size(-1)))

            s_forecam = cam_supp_feat * tmp_supp_pseudo_masks
            s_backcam = cam_supp_feat * (1 - tmp_supp_pseudo_masks)
            
            tmp_supp_pseudo_masks = tmp_supp_pseudo_masks.squeeze(1)      # [b,h,w]

            s_pcam_pmask = self.CossimCompute(s_forecam, supp_pcam_pmask4_pro)    # [b, h, w]
            s_ncam_pmask = self.CossimCompute(s_backcam, supp_ncam_pmask4_pro)    # [b, h, w]

            s_cam_mask = torch_minmaxnormalize(s_pcam_pmask, tmp_supp_pseudo_masks) + torch_minmaxnormalize(s_ncam_pmask, 1 - tmp_supp_pseudo_masks)

            # 对s_cam_mask，忽略其值最小的20%像素点重新进行01norm
            # s_cam_mask = norm_cam_mask(s_cam_mask)

            corr_supp = repeat(s_cam_mask, 'b h w -> b c h w', c=1)
            corr_supp = F.interpolate(corr_supp, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_supp_mask_list[i] = corr_supp

            if self.vis and TTA_vis:
                for b in range(bsz):

                    vis_path = os.path.join(self.args.logpath, self.args.vis_name, q_name[b])
                    if not os.path.exists(vis_path):
                        os.makedirs(vis_path, exist_ok=True)

                    tmp_corr_query = F.interpolate(corr_query.float(), size=(query_img.size(-2), query_img.size(-1)), mode='bilinear')
                    q_sim = tmp_corr_query[b, 0]

                    q_img = query_img[b]
                    q_vis_image = torch.zeros_like(q_img, dtype=torch.uint8)
                    for c in range(q_img.shape[0]):
                        q_img_c = q_img[c]
                        q_img_c = (q_img_c - q_img_c.min()) / (q_img_c.max() - q_img_c.min()) * 255
                        q_vis_image[c] = q_img_c.to(torch.uint8)
                    q_vis_image = np.int64(q_vis_image.detach().cpu())
                    q_vis_image[q_vis_image > 255] = 255
                    q_vis_image[q_vis_image < 0] = 0
                    q_vis_image = np.uint8(q_vis_image.transpose(1,2,0))
                    plt.imsave(os.path.join(vis_path, f"q_sim_heatmap.png"), generate_vis(t2n(q_sim), q_vis_image, color='cam'))

                    q_sim = repeat(q_sim*255, 'h w -> h w (repeat)', repeat=3)
                    q_sim = q_sim.detach().cpu().numpy().astype(np.uint8)
                    plt.imsave(os.path.join(vis_path, f"q_sim.png"), q_sim)

                    tmp_q_pcam_pmask = F.interpolate(repeat(torch_minmaxnormalize(q_pcam_pmask, tmp_query_pseudo_masks).float(), "b h w -> b c h w", c=1), size=(query_img.size(-2), query_img.size(-1)), mode='bilinear')
                    tmp_q_pcam_pmask = tmp_q_pcam_pmask[b, 0]
                    tmp_q_pcam_pmask = repeat(tmp_q_pcam_pmask*255, 'h w -> h w (repeat)', repeat=3)
                    tmp_q_pcam_pmask = tmp_q_pcam_pmask.detach().cpu().numpy().astype(np.uint8)
                    plt.imsave(os.path.join(vis_path, f"pcam_pmask.png"), tmp_q_pcam_pmask)

                    tmp_q_pcam_nmask = F.interpolate(repeat(torch_minmaxnormalize(q_pcam_nmask, tmp_query_pseudo_masks).float(), "b h w -> b c h w", c=1), size=(query_img.size(-2), query_img.size(-1)), mode='bilinear')
                    tmp_q_pcam_nmask = tmp_q_pcam_nmask[b, 0]
                    tmp_q_pcam_nmask = repeat(tmp_q_pcam_nmask*255, 'h w -> h w (repeat)', repeat=3)
                    tmp_q_pcam_nmask = tmp_q_pcam_nmask.detach().cpu().numpy().astype(np.uint8)
                    plt.imsave(os.path.join(vis_path, f"pcam_nmask.png"), tmp_q_pcam_nmask)

                    tmp_q_ncam_pmask = F.interpolate(repeat(torch_minmaxnormalize(q_ncam_pmask, 1 - tmp_query_pseudo_masks).float(), "b h w -> b c h w", c=1), size=(query_img.size(-2), query_img.size(-1)), mode='bilinear')
                    tmp_q_ncam_pmask = tmp_q_ncam_pmask[b, 0]
                    tmp_q_ncam_pmask = repeat(tmp_q_ncam_pmask*255, 'h w -> h w (repeat)', repeat=3)
                    tmp_q_ncam_pmask = tmp_q_ncam_pmask.detach().cpu().numpy().astype(np.uint8)
                    plt.imsave(os.path.join(vis_path, f"ncam_pmask.png"), tmp_q_ncam_pmask)

                    tmp_q_ncam_nmask = F.interpolate(repeat(torch_minmaxnormalize(q_ncam_nmask, 1 - tmp_query_pseudo_masks).float(), "b h w -> b c h w", c=1), size=(query_img.size(-2), query_img.size(-1)), mode='bilinear')
                    tmp_q_ncam_nmask = tmp_q_ncam_nmask[b, 0]
                    tmp_q_ncam_nmask = repeat(tmp_q_ncam_nmask*255, 'h w -> h w (repeat)', repeat=3)
                    tmp_q_ncam_nmask = tmp_q_ncam_nmask.detach().cpu().numpy().astype(np.uint8)
                    plt.imsave(os.path.join(vis_path, f"ncam_nmask.png"), tmp_q_ncam_nmask)

                    s_vis_image = torch.empty_like(tmp_supp_img[b], dtype=torch.uint8)
                    for c in range(tmp_supp_img.shape[1]):
                        s_img_c = tmp_supp_img[b, c]
                        s_img_c = (s_img_c - s_img_c.min()) / (s_img_c.max() - s_img_c.min()) * 255
                        s_vis_image[c] = s_img_c.to(torch.uint8)

                    # save results
                    s_vis_image = np.int64(s_vis_image.detach().cpu())
                    s_vis_image[s_vis_image > 255] = 255
                    s_vis_image[s_vis_image < 0] = 0
                    s_vis_image = np.uint8(s_vis_image.transpose(1,2,0))

                    tmp_supp_mask = t2n(supp_mask[b][0].long())

                    plt.imsave(ospj(vis_path, f"s_original_{s_names[i][b]}.png"), s_vis_image)
                    plt.imsave(ospj(vis_path, f"s_fore_{s_names[i][b]}.png"), generate_vis(tmp_supp_mask, s_vis_image, rate=0.4, color='b'))
                    plt.imsave(ospj(vis_path, f"s_mask_{s_names[i][b]}.png"), tmp_supp_mask*255)

                    tmp_corr_supp = F.interpolate(corr_supp.float(), size=(query_img.size(-2), query_img.size(-1)), mode='bilinear')
                    s_sim = tmp_corr_supp[b, 0]
                    s_sim = repeat(s_sim*255, 'h w -> h w (repeat)', repeat=3)
                    s_sim = s_sim.detach().cpu().numpy().astype(np.uint8)
                    plt.imsave(os.path.join(vis_path, f"s_sim{s_names[i][b]}.png"), s_sim)

        corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)      # [bsz, k, 30, 30]
        corr_query_mask = torch.clamp(corr_query_mask, 0, 1)
        
        with autocast('cuda', enabled=False):
            sim_measurement = F.binary_cross_entropy(F.interpolate(corr_query_mask, size=(query_mask.shape[-2], query_mask.shape[-1]), mode='bilinear', align_corners=True).squeeze(1), query_mask.float())   # [b, 473, 473], [b, 1, 30, 30]

        # Support Prototype
        supp_pro_list = rearrange(supp_pro_list, "k b c h -> b k c h")
        supp_pcam_pmask_list = rearrange(supp_pcam_pmask_list, "k b c h -> b k c h")
        supp_ncam_pmask_list = rearrange(supp_ncam_pmask_list, "k b c h -> b k c h")

        # weight_soft: [b, k, 1, 1]
        supp_pro = rearrange((weight_soft*supp_pro_list).sum(1, True), 'b h c w -> b c h w')     # [b, 1, c, 1]
        supp_pcam_pmask_pro = rearrange((weight_soft*supp_pcam_pmask_list).sum(1, True), 'b h c w -> b c h w')
        supp_ncam_pmask_pro = rearrange((weight_soft*supp_ncam_pmask_list).sum(1, True), 'b h c w -> b c h w')

        # Tile & Cat
        whole_pro_feat = supp_pro.expand_as(query_feat)

        # torch.Size([b, 1, h, w])
        pcam_pmask_pro_feat = supp_pcam_pmask_pro.expand_as(query_feat) # [bsz, 256, 30, 30]
        ncam_pmask_pro_feat = supp_ncam_pmask_pro.expand_as(query_feat) # [bsz, 256, 30, 30]        
        tmp_query_pseudo_masks = F.interpolate(query_pseudo_masks.float(), size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
        query_fore_pro_feat = pcam_pmask_pro_feat * tmp_query_pseudo_masks + ncam_pmask_pro_feat * (1-tmp_query_pseudo_masks)

        q_merge_feat = torch.cat([
            query_feat, 
            whole_pro_feat,
            query_fore_pro_feat,
            # pcam_pmask_pro_feat,
            # ncam_pmask_pro_feat,
            corr_query_mask], 1)   # 256*n+1
        q_merge_feat = self.init_merge(q_merge_feat.detach())       # [bsz, 256, 30, 30]
        query_meta_out = q_merge_feat

        supp_meta_out = torch.empty_like(decode_supp_feat, device=decode_supp_feat.device)      # [k, bsz, 256, 30, 30]

        supp_pro_list = rearrange(supp_pro_list, "b k c h -> k b c h")
        supp_pcam_pmask_list = rearrange(supp_pcam_pmask_list, "b k c h -> k b c h")
        supp_ncam_pmask_list = rearrange(supp_ncam_pmask_list, "b k c h -> k b c h")

        for i, (supp_feat, supp_pro, supp_pcam_pmask_pro, supp_ncam_pmask_pro, corr_supp_mask, supp_pseudo_mask) in enumerate(zip(decode_supp_feat, supp_pro_list, supp_pcam_pmask_list, supp_ncam_pmask_list, corr_supp_mask_list, supp_pseudo_masks_list)):
            '''
            supp_feat: decode_supp_feat, [k, bsz, 256, 30, 30]
            supp_pro: supp_pro_list, [k, bsz, 256, 1]
            supp_pcam_pmask_pro: supp_pcam_pmask_list, [k, bsz, 256, 1]
            supp_ncam_pmask_pro: supp_ncam_pmask_list, [k, bsz, 256, 1]
            corr_supp_mask: corr_supp_mask_list, [k, bsz, 1, 30, 30]
            supp_pseudo_mask: supp_pseudo_masks_list, [k, bsz, 1, 473, 473]
            '''

            # Tile & Cat
            whole_pro_feat = repeat(supp_pro, 'b c h -> b c h w', w=1).expand_as(supp_feat)

            pcam_pmask_pro_feat = repeat(supp_pcam_pmask_pro, 'b c h -> b c h w', w=1).expand_as(supp_feat) # [bsz, 256, 30, 30]
            ncam_pmask_pro_feat = repeat(supp_ncam_pmask_pro, 'b c h -> b c h w', w=1).expand_as(supp_feat) # [bsz, 256, 30, 30]     

            tmp_supp_pseudo_masks = F.interpolate(supp_pseudo_mask.float(), size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
            corr_supp_mask = F.interpolate(corr_supp_mask.float(), size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
            supp_fore_pro_feat = pcam_pmask_pro_feat * tmp_supp_pseudo_masks + ncam_pmask_pro_feat * (1-tmp_supp_pseudo_masks)

            s_merge_feat = torch.cat([
                supp_feat, 
                whole_pro_feat,
                supp_fore_pro_feat,
                # pcam_pmask_pro_feat,
                # ncam_pmask_pro_feat,
                corr_supp_mask], 1)   # 256*n+1
            s_merge_feat = self.init_merge(s_merge_feat.detach())   # [bsz, 256, 30, 30]
            supp_meta_out[i] = s_merge_feat

        supp_meta_out_list = supp_meta_out

        rectified_query_mask = F.interpolate(query_mask.unsqueeze(1).float(), size=(supp_feat.size(2), supp_feat.size(3)), mode='nearest').bool() if self.training else None
        # rectified_query_mask = None
        rectified_supp_masks = [F.interpolate(supp_mask_list[j].float(), size=(supp_feat.size(2), supp_feat.size(3)), mode='nearest').bool() for j in range(self.shot)]

        S_middle_out, Q_middle_out = [], []

        for i in range(self.block_num):
            query_meta = self.Q_block[i](query_meta_out)   # 1080->256

            _, c, h, w = query_meta.shape

            # supp merge feat
            supp_meta_in_list = supp_meta_out_list.clone()
            query_meta_list = torch.empty((self.shot, bsz, c, h, w), device=supp_meta_in_list.device)    # shot length, # [bsz, 256]

            S_shot_middle_out = []
            for j in range(self.shot):
                supp_meta = supp_meta_in_list[j]
                supp_meta = self.S_block[i](supp_meta)    # 1080->256


                if i < self.cross_num:

                    # compute supp & query out
                    # supp_mask_list: shot length, torch.Size([bz, 1, 473, 473])
                    tmp_query_meta, mask_ratio = self.CrossAttn[i](x=query_meta, x_s=supp_meta, ym_s=rectified_supp_masks[j], ym=rectified_query_mask)
                    mask_ratio_list[i, j, :] = mask_ratio

                    query_meta_list[j] = tmp_query_meta

                supp_meta_out_list[j] = supp_meta

                if i < self.block_num - 1:      # the last decoder block is not included
                    S_shot_middle_out.append(self.S_cls_head[i](supp_meta))

            if i < self.cross_num:
                query_meta = query_meta_list.mean(dim=0)

            query_meta_out = self.Q_block_res2_meta[i](query_meta) + query_meta

            if i < self.block_num - 1:      # the last decoder block is not included
                Q_middle_out.append(self.Q_cls_head[i](query_meta))
                
            if S_shot_middle_out:
                S_middle_out.append(torch.stack(S_shot_middle_out, dim=0))    # [s, b, c, h, w] or []

        query_meta_out = self.Q_cls_meta(query_meta_out)

        supp_meta_out = []     
        for i in range(self.shot):
            supp_meta = supp_meta_out_list[i]
            supp_meta = self.S_cls_meta(supp_meta)
            supp_meta_out.append(supp_meta)

        return query_meta_out, supp_meta_out, rectified_query_mask, \
            rectified_supp_masks, torch.stack(S_middle_out, dim=0), torch.stack(Q_middle_out, dim=0), sim_measurement,  mask_ratio_list.mean(dim=1)

    def CossimCompute(self, q_feats: torch.Tensor, feat_prototype: torch.Tensor):
        """
        q_feats: torch.Tensor in shape [b, c, h, w]
        prototype: torch.Tensor in shape [b, c]

        return torch.Tensor in shape [b, h, w]
        """
        eps = 1e-5
        b, c, h, w = q_feats.size()
        q_feats = rearrange(q_feats, "b c h w -> b c (h w)")       # [b, c, hw]
        q_feats_norm = q_feats / (q_feats.norm(dim=1, p=2, keepdim=True) + eps)
        
        prototype_norm = feat_prototype.unsqueeze(2)
        prototype_norm = prototype_norm / (prototype_norm.norm(dim=1, p=2, keepdim=True) + eps)     # [b, c, 1]
        
        cossim = repeat(torch.bmm(rearrange(q_feats_norm, 'b c d -> b d c'), prototype_norm).squeeze(2), 'b (h w) -> b h w', h=h, w=w)
        return cossim