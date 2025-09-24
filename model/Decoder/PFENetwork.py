from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl


class Decoder(pl.LightningModule):
    '''
    upsampling decoder, only using top layer
    '''
    def __init__(self, ppm_scales=[64, 32, 16, 8]):
        super(Decoder, self).__init__()

        reduce_dim = 256
        fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, 2, kernel_size=1)
        )  

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5)             
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5)               
        )

        self.pyramid_bins = ppm_scales
        
        self.beta_conv = nn.ModuleList()
        self.inner_cls = nn.ModuleList()
        for _ in self.pyramid_bins:
 
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, 2, kernel_size=1)
            ))            


        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

    def forward(self, feats: List[torch.Tensor], sz: int):
        '''
        feats in shape List([1, 256, 100, 100] [1, 256, 50, 50] [1, 256, 25, 25] [1, 256, 13, 13])
        '''

        h, w = sz

        pyramid_feat_list = []  # 25*25
        out_list = []   # [64, 32, 16, 8]
        feat = feats[0]
        
        for idx, bin in enumerate(self.pyramid_bins):   # [64, 32, 16, 8]

            merge_feat_bin = F.interpolate(feat, size=(bin, bin), mode='bilinear', align_corners=True)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()     # 梯度会叠加
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(feat.size(2), feat.size(3)), mode='bilinear', align_corners=True)  # all resize to largest size
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, dim=1)    # 256*4=1024 channels
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat           
        out = self.cls(query_feat)
        
        #   Output Part
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out, out_list

        