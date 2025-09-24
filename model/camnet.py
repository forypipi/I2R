from functools import reduce
from math import sqrt
from operator import add
from pathlib import Path
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from torchvision.models import resnet, ResNet50_Weights, ResNet101_Weights
from torchvision.models import ResNet101_Weights

from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import Normalize

from einops import rearrange, repeat

from model.base.feature import extract_feat_res
from model.base.utils import get_gram_matrix
from model.Middle import Middle
from model.Decoder import PFENet_decoder, SETRNaive_decoder, SETRPUP_decoder, SETRMLA_decoder, Segformer_decoder, Segmentor_decoder
from model.BaseModel.PSPNet import BaseModel as PSPNet

from common.evaluation import AverageMeter, Evaluator

import os, sys

from CAM_Generator.CAM_generator import SCRIPT_DIR, Generator_CAM
from CAM_Generator.base.utils import torch_minmaxnormalize, torch_zscorenormalize
from data.dataset import FSSDatasetModule
from CAM_Generator.attribution  import CAMWrapper, GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, ScoreCAM, LayerCAM, AblationCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
from CAM_Generator.attribution.utils import normalize_saliency, visualize_single_saliency

from crf import CRF

import torch.nn.functional as F

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]

class CamNetwork(pl.LightningModule):
    average_meter: Dict[str,AverageMeter]
    def __init__(self, args, use_original_imgsize):
        super(CamNetwork, self).__init__()

        # 1. Backbone network initialization
        self.args = args
        self.start_blocks = 2       # extrac block 'self.start_blocks' to 'self.end_blocks'(both include)
        self.end_blocks = 3
        self.backbone_type = self.args.backbone
        self.use_original_imgsize = use_original_imgsize

        self.cam_dict = {
            'BagCAM': BagCAM,
            'GradCAM': GradCAM,
            'GradCAMPlusPlus': GradCAMPlusPlus, 
            'XGradCAM': XGradCAM, 
            'ScoreCAM': ScoreCAM, 
            'LayerCAM': LayerCAM, 
            'AblationCAM': AblationCAM, 
            'FullGrad': FullGrad, 
            'EigenCAM': EigenCAM, 
            'EigenGradCAM': EigenGradCAM, 
            'HiResCAM': HiResCAM
        }

        if self.args.backbone == 'resnet50':
                
            self.backbone = resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.layer0 = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1, 
                self.backbone.relu,
                self.backbone.maxpool
            )

            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 6, 3]
            self.feat_ids = list(range(sum(self.nbottlenecks[:self.start_blocks])+1, sum(self.nbottlenecks[:self.end_blocks])+1))
            self.low_fea_id = sum(self.nbottlenecks[:self.args.low_fea])

        elif self.args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 23, 3]
            self.feat_ids = list(range(sum(self.nbottlenecks[:self.start_blocks])+1, sum(self.nbottlenecks)+1))
            self.low_fea_id = sum(self.nbottlenecks[:self.args.low_fea])
        else:
            raise Exception(f'Unavailable backbone: {self.args.backbone}')

        # loading BAM Base Learner
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = root_dir / 'BAM_Pretrained' / self.args.benchmark / self.args.backbone / f'fold{self.args.fold}' / 'best_model.ckpt'
        checkpoint = torch.load(weight_path)
        checkpoint["state_dict"] = {k.replace('BaseLearner.', ''): v for k, v in checkpoint["state_dict"].items() if k.startswith("BaseLearner.")}
        self.learner_base = PSPNet(backbone=self.backbone, args=self.args)
        self.learner_base.load_state_dict(checkpoint["state_dict"])
        self.backbone.layer0 = self.learner_base.layer0
        self.backbone.layer1 = self.learner_base.layer1
        self.backbone.layer2 = self.learner_base.layer2
        self.backbone.layer3 = self.learner_base.layer3
        self.backbone.layer4 = self.learner_base.layer4

        for param in self.learner_base.ppm.parameters():
            param.requires_grad = False
        for param in self.learner_base.cls.parameters():
            param.requires_grad = False

        self.datapath = self.args.datapath

        self.vis = self.args.vis

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.learner_base.eval()

        self.cam_net = self.cam_dict[self.args.method](self.backbone)        
        self.channels = [256, 512, 1024, 2048]
    
    
        self.CAM2Mask = CRF(n_ref=3, n_out=2, trainable_kstd=True, 
                   sxy_bf=1,
                   sc_bf=1,
                   compat_bf=1,
                   sxy_spatial=1,
                   compat_spatial=1,
                   num_iter=5, device=next(self.backbone.parameters()).device)
        self.CAM2Mask.load_state_dict(torch.load(Path(SCRIPT_DIR) / "pretrain_model" / "crfasrnn_weights.pth", map_location=next(self.backbone.parameters()).device))


        if self.args.MAP_feats == "all":
            if self.args.area_num == 4:
                expand_feat = 6
            else:
                expand_feat = 4
        else:
            expand_feat = 2

        if self.args.similar_map == 'all':
            self.middle = Middle(
                self.nbottlenecks[self.start_blocks:self.end_blocks], 
                self.channels[self.start_blocks:self.end_blocks],
                expand_feat=expand_feat,
                out_ch=self.args.channel
                )
        else:
            self.middle = Middle(
                [1, 1, 1, 1][self.start_blocks:self.end_blocks], 
                self.channels[self.start_blocks:self.end_blocks],
                expand_feat=expand_feat,
                out_ch=self.args.channel
                )

        if self.args.decoder == "PFENet":
            self.decoder = PFENet_decoder()        
        elif self.args.decoder == "SETR-Naive":
            self.decoder = SETRNaive_decoder()
        elif self.args.decoder == "SETR-PUP":
            self.decoder = SETRPUP_decoder()
        elif self.args.decoder == "SETR-MLA":
            self.decoder = SETRMLA_decoder()
        elif self.args.decoder == "SegFormer":
            self.decoder = Segformer_decoder()
        elif self.args.decoder == "segmentor":
            self.decoder = Segmentor_decoder()
        else:
            raise Exception(f"Wrong args decoder: {self.args.decoder}")


        # for BAM post-processing
        self.base_classes = 15 if self.args.benchmark=='pascal' else 60

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



        # for example input test
        self.cross_entropy_loss = F.cross_entropy

        dm = FSSDatasetModule(args)
        transform_unnorm = transforms.Compose([transforms.Resize(size=(self.args.img_size, self.args.img_size)),
                                                transforms.ToTensor()])

        transform_norm = transforms.Compose([transforms.Resize(size=(self.args.img_size, self.args.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STDDEV)])
        self.example_input_array = [next(iter(DataLoader(
            dm.datasets[dm.args.benchmark](
            dm.args.datapath,
            fold=dm.args.fold,
            transform_norm=transform_norm,
            transform_unnorm=transform_unnorm,
            split='trn',
            shot=self.args.shot,
            use_original_imgsize=False),
            batch_size=self.args.trn_bsz,
            shuffle=True)
            ))]

        self.max_miou, self.max_fbiou = 0, 0

    def forward(self, batch: dict):
        '''
        query_img: torch.Tensor, [b, c, h, w]
        query_mask: torch.Tensor, [b, h, w]
        query_base_mask: torch.Tensor, [b, h, w]        
        query_name: list, [b,]
        support_imgs: torch.Tensor, [b, k, c, h, w]
        support_masks: torch.Tensor, [b, k, h, w]
        support_imgfores: torch.Tensor, [b, k, c, h, w]
        support_names: list, [1, k, b,]
        class_sample_idx: list, [b,]
        '''

        b, k, c, h, w = batch['support_imgs'].shape
        support_img = rearrange(batch['support_imgs'], 'b k c h w -> (b k) c h w')
        support_mask = repeat(batch['support_masks'], 'b k h w -> (b k) c h w', c=1)
        query_img = batch['query_img']
        support_names = batch["support_names"]
        query_names = batch["query_name"]
        cat_idx = batch['class_sample_idx']

        with torch.no_grad():

            query_feats, _, _ = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats, img_fore_prob, img_fore_cls = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids, pool=True, support_mask=support_mask)

            query_target_feats = [query_feats[i-1] for i in self.feat_ids]
            support_target_feats = [support_feats[i-1] for i in self.feat_ids]

            query_low_layer = query_feats[self.low_fea_id]
            support_low_layer = repeat(support_feats[self.low_fea_id], '(b k) c h w -> b k c h w', k=k)
            support_low_layer = rearrange(support_low_layer, 'b k c h w -> k b c h w')

            with torch.enable_grad():
                # _ _ _ _ torch.Size([b, k, h, w]) torch.Size([b, k, h, w]) torch.Size([b, h, w])
                _, _, support_masks, support_pseudo_masks, query_cam_masks = Generator_CAM(
                    cam_net = self.cam_net,
                    datapath=self.datapath,
                    batch=batch,
                    vis=self.args.vis,
                    later_mask=self.args.later_mask,
                    img_fore_cls=img_fore_cls,
                    vis_path=os.path.join(self.args.logpath, "vis"),
                    threshold=self.args.threshold,
                    CAM2Mask=self.CAM2Mask
                    )

            support_masks = support_masks.bool().clone().detach()
            query_cam_masks = query_cam_masks.unsqueeze(1).bool().clone().detach()
            support_pseudo_masks = support_pseudo_masks.clone().detach()


            poscam_posmask = torch.logical_and(support_masks, support_pseudo_masks)
            negcam_posmask = torch.logical_and(support_masks,torch.logical_not(support_pseudo_masks))
            poscam_negmask = torch.logical_and(torch.logical_not(support_masks),support_pseudo_masks)
            negcam_negmask = torch.logical_and(torch.logical_not(support_masks), torch.logical_not(support_pseudo_masks))

            original_h, original_w = poscam_posmask.size(-2), poscam_posmask.size(-1)

            q_originalsize_sims, feats = [], []
            first = True        # for collect q_sims using tensor cat instead using list

            # q: torch.Size([bk, 256, 100, 100])*3 -> torch.Size([bk, 512, 50, 50])*4 -> torch.Size([bk, 1024, 25, 25])*6 -> torch.Size([bk, 2048, 13, 13])*3
            for i, (q, s) in enumerate(zip(query_target_feats, support_target_feats)):
                
                q, s = q.detach(), s.detach()
                
                target_layer = np.cumsum(self.nbottlenecks[self.start_blocks:self.end_blocks]).tolist()       # [3, 4 ,6, 3] to [3, 7, 13, 16]
                if i+1 in target_layer:
                    record = True
                else:
                    record = False

                tmp_poscam_posmask = F.interpolate(poscam_posmask.float(), size=(s.size(-2), s.size(-1)))   # in shape [b, k, h, w]
                tmp_negcam_posmask = F.interpolate(negcam_posmask.float(), size=(s.size(-2), s.size(-1)))
                tmp_poscam_negmask = F.interpolate(poscam_negmask.float(), size=(s.size(-2), s.size(-1)))
                tmp_negcam_negmask = F.interpolate(negcam_negmask.float(), size=(s.size(-2), s.size(-1)))
                tmp_mask = F.interpolate(support_masks.float(), size=(s.size(-2), s.size(-1)))

                tmp_query_cam_masks = F.interpolate(query_cam_masks.float(), size=(q.size(-2), q.size(-1)))
                
                tmp_s = rearrange(s, '(b k) c h w -> b k c h w', k=k)

                proto_poscam_posmask = self.MaskedAveragePooling(tmp_s, tmp_poscam_posmask)    # in shape [b, c]
                proto_negcam_posmask = self.MaskedAveragePooling(tmp_s, tmp_negcam_posmask)
                proto_poscam_negmask = self.MaskedAveragePooling(tmp_s, tmp_poscam_negmask)
                proto_negcam_negmask = self.MaskedAveragePooling(tmp_s, tmp_negcam_negmask)
                s_MAP = self.MaskedAveragePooling(tmp_s, tmp_mask) # in shape [b, c]


                q_forecam = q * tmp_query_cam_masks
                q_backcam = q * (1 - tmp_query_cam_masks)

                tmp_query_cam_masks = F.interpolate(query_cam_masks.float(), size=(q.size(-2), q.size(-1)))
                tmp_query_cam_masks = tmp_query_cam_masks.squeeze(1)

                q_poscam_posmask = self.CossimCompute(q_forecam, proto_poscam_posmask)    # [b, h, w]
                q_poscam_negmask = self.CossimCompute(q_forecam, proto_poscam_negmask)    # [b, h, w]
                q_negcam_posmask = self.CossimCompute(q_backcam, proto_negcam_posmask)    # [b, h, w]
                q_negcam_negmask = self.CossimCompute(q_backcam, proto_negcam_negmask)    # [b, h, w]

                if self.args.area_num == 4:
                    q_forecam_prob = torch.softmax(torch.stack([q_poscam_posmask, q_poscam_negmask], dim=1), dim=1)[:,0]    # [b, h, w]
                    q_forecam_prob = q_forecam_prob * tmp_query_cam_masks
                    q_backcam_prob = torch.softmax(torch.stack([q_negcam_posmask, q_negcam_negmask], dim=1), dim=1)[:,0]   # [b, h, w]
                    q_backcam_prob = q_backcam_prob * (1 - tmp_query_cam_masks)
                    q_cam_mask = q_forecam_prob + q_backcam_prob     # [b, h, w]
                else:
                    q_cam_mask = q_poscam_posmask + q_negcam_posmask

                q_cam_mask = torch_minmaxnormalize(q_cam_mask)       # [b, h, w]

                if self.args.sim_norm == "ZNorm":
                    tmp_q_cam_mask = torch_zscorenormalize(q_cam_mask)
                else:
                    tmp_q_cam_mask = q_cam_mask.clone().detach()

                if self.args.similar_map == 'all':
                # using all similar map in same layer
                    if first:
                        q_sims = repeat(tmp_q_cam_mask, 'b h w -> b c h w', c=1)
                        first= False
                    else:
                        q_sims = torch.cat([q_sims, repeat(tmp_q_cam_mask, 'b h w -> b c h w', c=1)], dim=1)
                else:
                # using only 1 similar map in same layer
                    q_sims = repeat(tmp_q_cam_mask, 'b h w -> b c h w', c=1)

                MAP_feat = repeat(s_MAP, 'b c -> b c h w', h=q.size(-2), w=q.size(-1))
                poscam_posmask_feat = repeat(proto_poscam_posmask, 'b c -> b c h w', h=q.size(-2), w=q.size(-1))     # [b, c, h, w]
                poscam_negmask_feat = repeat(proto_poscam_negmask, 'b c -> b c h w', h=q.size(-2), w=q.size(-1))     # [b, c, h, w]
                negcam_posmask_feat = repeat(proto_negcam_posmask, 'b c -> b c h w', h=q.size(-2), w=q.size(-1))     # [b, c, h, w]
                negcam_negmask_feat = repeat(proto_negcam_negmask, 'b c -> b c h w', h=q.size(-2), w=q.size(-1))     # [b, c, h, w]

                if record:
                    if self.args.MAP_feats == "all":
                        if self.args.area_num == 4:
                            feats.append(torch.cat([
                                q, 
                                MAP_feat, 
                                poscam_posmask_feat, 
                                poscam_negmask_feat, 
                                negcam_posmask_feat, 
                                negcam_negmask_feat, 
                                q_sims], dim=1).clone().detach())
                        else:
                            feats.append(torch.cat([
                                q, 
                                MAP_feat, 
                                poscam_posmask_feat, 
                                negcam_posmask_feat, 
                                q_sims], dim=1).clone().detach())
                    else:
                        feats.append(torch.cat([q, MAP_feat, q_sims], dim=1).clone().detach())
                    first = True

                if self.args.vis:
                    tmp_q_cam_mask = repeat(q_cam_mask, 'b h w -> b c h w', c=1).clone().detach()
                    tmp_q_cam_mask = F.interpolate(tmp_q_cam_mask, size=(original_h, original_w), mode='bilinear').squeeze(1)
                    q_originalsize_sims.append(tmp_q_cam_mask)
                    
            if self.args.vis:
                vis_path = os.path.join(self.args.logpath, "vis")

                for i, q_sim_mask in enumerate(q_originalsize_sims):
                    for j, (query_name, q_sim) in enumerate(zip(query_names, q_sim_mask)):

                        path = os.path.join(vis_path, str(j), f"{query_name}")
                        if not os.path.exists(path):
                            os.mkdir(path)

                        q_sim = repeat(q_sim*255, 'h w -> h w (repeat)', repeat=3)
                        q_sim = q_sim.cpu().numpy().astype(np.uint8)
                        
                        plt.imsave(os.path.join(path, f"sim_{i}.png"), q_sim)
        

        feats = self.middle(feats)  # [1, 256, 100, 100] [1, 256, 50, 50] [1, 256, 25, 25] [1, 256, 13, 13]

        # BAM post-processing
        shared_masks, inner_masks = self.decoder(feats, sz=(original_h, original_w))     # [B, 2, H, W], list of [B, 2, H, W]
        feats.clear()

        # K-Shot Reweighting
        que_gram = get_gram_matrix(query_low_layer) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in support_low_layer:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(b,1,1,1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.args.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight:torch.Tensor = self.kshot_rw(val1)       # channel shot->2->shot
            weight = weight.gather(1, idx2)                 # dim=1, index=idx2
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True) # [bs, 1, 1, 1]            


        meta_out = shared_masks
        base_out = self.learner_base(query_img)
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:,:1,:,:]   # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:,1:,:,:]

        if self.training:
            # only other background prompt current foreground during training, foreground cannot prompt itself
            c_id_array = torch.arange(self.base_classes+1, device='cuda')       # 16 for pascal
            base_map_list = []
            for b_id in range(b):      # batch_size
                c_id = cat_idx[0][b_id] + 1     # selected class in batch
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)     # background(0) and base selected class is False, else is True
                base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list,0)
            # <alternative implementation>
            # gather_id = (cat_idx[0]+1).reshape(bs,1,1,1).expand_as(base_out_soft[:,0:1,:,:]).cuda()
            # fg_map = base_out_soft.gather(1,gather_id)
            # base_map = base_out_soft[:,1:,:,:].sum(1,True) - fg_map            
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)

        est_map = est_val.expand_as(meta_map_fg)        # [bs, 1, 1, 1] to [bs, 1, 60, 60]

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))  # [bs, 2, 60, 60] to [bs, 1, 60, 60]
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        return final_out, meta_out, base_out, inner_masks


    def MaskedAveragePooling(self, s_feats: torch.Tensor, area: torch.Tensor)->torch.Tensor:
        """
        s_feats: torch.Tenor in shape [b, k, c, h, w]
        area: mask area, torch.Tenor in shape [b, k, h, w]

        return feature's prototype in area, torch.Tensor in shape [b, c]
        """
        e = 1e-5
        area = repeat(area, 'b k h w -> b k (c) h w', c=s_feats.size(-3))
        mask_area = torch.sum(area, dim=(3, 4)) + e
        result = torch.sum(s_feats * area, dim=(3, 4)) / mask_area
        result = torch.mean(result, dim=1)
        return result

    def CossimCompute(self, q_feats: torch.Tensor, prototype: torch.Tensor):
        """
        q_feats: torch.Tensor in shape [b, c, h, w]
        prototype: torch.Tensor in shape [b, c]

        return torch.Tensor in shape [b, h, w]
        """
        eps = 1e-5
        b, c, h, w = q_feats.size()
        q_feats = rearrange(q_feats, "b c h w -> b c (h w)")       # [b, c, hw]
        q_feats_norm = q_feats / (q_feats.norm(dim=1, p=2, keepdim=True) + eps)
        prototype = prototype.unsqueeze(2)
        prototype_norm = prototype / (prototype.norm(dim=1, p=2, keepdim=True) + eps)
        
        cossim = torch.bmm(q_feats_norm.transpose(1, 2), prototype_norm).view(b, h, w)
        return cossim

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1
        return pred_mask

    def compute_objective(self, logit_mask: torch.Tensor, gt_mask: torch.Tensor):
        if self.training:
            gt_mask = F.interpolate(gt_mask, size=(logit_mask.shape[-2], logit_mask.shape[-1]), mode='nearest')
            logit_mask = rearrange(logit_mask, "b c h w -> b c (h w)")
            gt_mask = gt_mask.squeeze(1)
            gt_mask = rearrange(gt_mask.long(), "b h w -> b (h w)")
        else:
            logit_mask = F.interpolate(logit_mask, size=(gt_mask.shape[-2], gt_mask.shape[-1]), mode='bilinear')
            logit_mask = rearrange(logit_mask, "b c h w -> b c (h w)")
            gt_mask = gt_mask.squeeze(1)
            gt_mask = rearrange(gt_mask.long(), "b h w -> b (h w)")

        return self.cross_entropy_loss(logit_mask, gt_mask, ignore_index=255)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
    
    def training_step(self, batch, batch_idx):

        target = batch['query_mask']
        bsz = target.size(0)
        logit_mask, meta_mask, base_mask, inner_masks = self.forward(batch)
        main_loss = self.compute_objective(logit_mask, repeat(target, 'b h w -> b c h w', c=1))
        meta_loss = self.compute_objective(meta_mask, repeat(target, 'b h w -> b c h w', c=1))
        
        deep_aux_loss = 0
        for deep_mask in inner_masks:
            deep_aux_loss += self.compute_objective(deep_mask, repeat(target, 'b h w -> b c h w', c=1))
        deep_aux_loss = deep_aux_loss / len(inner_masks)
        loss = main_loss + deep_aux_loss + meta_loss

        logit_mask = F.interpolate(logit_mask, size=(target.shape[-2], target.shape[-1]), mode='bilinear')
        logit_mask = logit_mask.max(1)[1]

        with torch.no_grad():
            intersection, union, target = Evaluator.classify_prediction(logit_mask, target, K=2, ignore_index=255)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()

            self.average_meter = self.train_average_meter if self.training else self.val_average_meter
            self.average_meter['intersection_meter'].update(intersection)
            self.average_meter['union_meter'].update(union)
            self.average_meter['target_meter'].update(target)

            self.average_meter['main_loss_meter'].update(main_loss.item(), bsz)
            self.average_meter['aux_loss_meter1'].update(meta_loss.item(), bsz)
            self.average_meter['deep_aux_loss_meter'].update(deep_aux_loss.item(), bsz)
            self.average_meter['loss_meter'].update(loss.item(), bsz)

        return loss

    def on_train_epoch_end(self):
        self.average_meter = self.train_average_meter
        avg_loss = self.average_meter['loss_meter'].avg
        avg_mainloss = self.average_meter['main_loss_meter'].avg
        avg_aux1loss = self.average_meter['aux_loss_meter1'].avg
        avg_deeploss = self.average_meter['deep_aux_loss_meter'].avg

        iou_class = self.average_meter['intersection_meter'].sum / (self.average_meter['union_meter'].sum + 1e-10)
        mIoU = np.mean(iou_class) * 100

        print(f'\nTrain result at epoch [{self.current_epoch}/{self.args.niter}]: mIoU {mIoU:.4f}, loss {avg_loss:.4f}.')

        dict = {
            f'trn/loss': avg_loss,
            f'trn/MainLoss': avg_mainloss,
            f'trn/MetaLoss': avg_aux1loss,
            f'trn/DeepLoss': avg_deeploss,
            f'trn/miou': mIoU.item(),
            }

        for k, v in dict.items():
            self.log(k, v, on_epoch=True, logger=True, sync_dist=True)


    def on_validation_start(self):
        if self.args.benchmark == 'pascal':
            self.split_gap = 5
        elif self.args.benchmark == 'coco':
            self.split_gap = 20
        self.class_intersection_meter = [0]*self.split_gap
        self.class_union_meter = [0]*self.split_gap  
        self.class_intersection_meter_m = [0]*self.split_gap
        self.class_union_meter_m = [0]*self.split_gap  
        self.class_intersection_meter_b = [0]*self.split_gap*3
        self.class_union_meter_b = [0]*self.split_gap*3
        self.class_target_meter_b = [0]*self.split_gap*3

    def validation_step(self, batch, batch_idx):
        self.average_meter = self.train_average_meter if self.training else self.val_average_meter
        bsz = batch['query_mask'].size(0)

        logit_mask, meta_mask, base_mask, inner_masks = self.forward(batch)

        # following BAM
        ori_label = batch['query_mask']
        ori_label_b = batch['query_base_mask']
        longerside = max(ori_label.size(1), ori_label.size(2))
        backmask = torch.ones(ori_label.size(0), longerside, longerside, device=batch['query_mask'].device)*255
        backmask_b = torch.ones(ori_label.size(0), longerside, longerside, device=batch['query_mask'].device)*255
        backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
        backmask_b[0, :ori_label.size(1), :ori_label.size(2)] = ori_label_b
        target = backmask.clone().long()
        target_b = backmask_b.clone().long()
        
        loss = self.compute_objective(logit_mask, repeat(target, 'b h w -> b c h w', c=1))

        logit_mask = F.interpolate(logit_mask, size=(target.shape[-2], target.shape[-1]), mode='bilinear', align_corners=True)
        meta_mask = F.interpolate(meta_mask, size=(target.shape[-2], target.shape[-1]), mode='bilinear', align_corners=True)
        base_mask = F.interpolate(base_mask, size=(target.shape[-2], target.shape[-1]), mode='bilinear', align_corners=True)

        logit_mask = logit_mask.max(1)[1]
        meta_mask = meta_mask.max(1)[1]
        base_mask = base_mask.max(1)[1]
        subcls = batch['class_sample_idx'][0].cpu().numpy()[0]

        intersection, union, new_target = Evaluator.classify_prediction(logit_mask, target, K=2, ignore_index=255)
        intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
        self.average_meter['intersection_meter'].update(intersection)
        self.average_meter['union_meter'].update(union)
        self.average_meter['target_meter'].update(new_target)
        self.class_intersection_meter[subcls] += intersection[1]
        self.class_union_meter[subcls] += union[1] 
        
        intersection, union, new_target = Evaluator.classify_prediction(meta_mask, target, K=2, ignore_index=255)
        intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
        self.average_meter['intersection_meter_m'].update(intersection)
        self.average_meter['union_meter_m'].update(union)
        self.average_meter['target_meter_m'].update(new_target)
        self.class_intersection_meter_m[subcls] += intersection[1]
        self.class_union_meter_m[subcls] += union[1]

        intersection, union, new_target = Evaluator.classify_prediction(base_mask, target_b, K=self.split_gap*3+1, ignore_index=255)
        intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
        for idx in range(1,len(intersection)):
            self.class_intersection_meter_b[idx-1] += intersection[idx]
            self.class_union_meter_b[idx-1] += union[idx]
            self.class_target_meter_b[idx-1] += new_target[idx]

        self.average_meter['loss_meter'].update(loss.item())


    def on_validation_epoch_end(self):
        self.average_meter = self.train_average_meter if self.training else self.val_average_meter

        iou_class = self.average_meter['intersection_meter'].sum / (self.average_meter['union_meter'].sum + 1e-10)
        iou_class_m = self.average_meter['intersection_meter_m'].sum / (self.average_meter['union_meter_m'].sum + 1e-10)
        mIoU = np.mean(iou_class) * 100
        mIoU_m = np.mean(iou_class_m) * 100
        
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

        class_miou = class_miou*100 / len(self.class_intersection_meter)
        class_miou_m = class_miou_m*100 / len(self.class_intersection_meter)
        class_miou_b = class_miou_b*100 / (len(self.class_intersection_meter_b) - len(target_b[target_b==0]))  # filter the results with GT mIoU=0

        print(f'\nmeanIoU---Val result: mIoU_final {class_miou:.4f}.')
        print(f'meanIoU---Val result: mIoU_meta {class_miou_m:.4f}.')   # meta
        print(f'meanIoU---Val result: mIoU_base {class_miou_b:.4f}.')   # base

        print('<<<<<<< Novel Results <<<<<<<')
        for i in range(self.split_gap):
            print(f'Class_{i+1} Result: iou_f {class_iou_class[i]*100:.4f}.')
            print(f'Class_{i+1} Result: iou_m {class_iou_class_m[i]*100:.4f}.')

        print('<<<<<<< Base Results <<<<<<<')
        for i in range(self.split_gap*3):
            if self.class_target_meter_b[i] == 0:
                print(f'Class_{i+1+self.split_gap} Result: iou_b None.')
            else:
                print(f'Class_{i+1+self.split_gap} Result: iou_b {class_iou_class_b[i]*100:.4f}.')

        print(f'FBIoU---Val result: FBIoU_f {mIoU:.4f}.')
        print(f'FBIoU---Val result: FBIoU_m {mIoU_m:.4f}.')
        for i in range(2):
            print(f'Class_{i} Result: iou_final {iou_class[i]*100:.4f}.')
            print(f'Class_{i} Result: iou_meta {iou_class_m[i]*100:.4f}.')
        print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        dict = {f'val/loss': self.average_meter['loss_meter'].avg,
                f'val/miou': class_miou,
                f'val/fb_iou': mIoU.item(),
                }
        
        if not self.training:
            self.max_miou: torch.Tensor = class_miou if class_miou>=self.max_miou else self.max_miou
            self.max_fbiou: torch.Tensor  = mIoU if mIoU>=self.max_fbiou else self.max_fbiou
            dict[f'val/max_miou'] = self.max_miou.item()
            dict[f'val/max_fb_iou'] = self.max_fbiou.item()

        for k, v in dict.items():
            self.log(k, v, on_epoch=True, logger=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        pred_mask = self.predict_mask_nshot(batch, self.args.shot)
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        self.average_meter.update(area_inter.cpu(), area_union.cpu(), batch['class_id'].cpu(), loss=None)

    def test_epoch_end(self, test_step_outputs):
        miou, fb_iou = self.average_meter.compute_iou()
        length = 16
        dict = {'benchmark'.ljust(length): self.args.benchmark,
                'fold'.ljust(length): self.args.fold,
                'test/miou'.ljust(length): miou.item(),
                'test/fb_iou'.ljust(length): fb_iou.item()}

        for k in dict:
            self.log(k, dict[k], on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.args.shot > 1:
            optimizer = torch.optim.SGD(
                [    
                    {'params': self.middle.parameters()},
                    {'params': self.decoder.parameters()},
                    {'params': self.gram_merge.parameters()},
                    {'params': self.cls_merge.parameters()},
                    {'params': self.kshot_rw.parameters()},
                ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [
                    {'params': self.middle.parameters()},
                    {'params': self.decoder.parameters()},
                    {'params': self.gram_merge.parameters()},
                    {'params': self.cls_merge.parameters()},
                ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        lr_lambda = lambda epoch: (1 - float(epoch) / self.args.niter) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler},}


    def get_progress_bar_dict(self):
        # to stop to show the version number in the progress bar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

if __name__=="__main__":
    pass
