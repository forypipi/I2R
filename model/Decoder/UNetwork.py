from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl


class Decoder(pl.LightningModule):
    '''
    upsampling decoder
    '''
    def __init__(self, channels: List, block_num: int=1, in_ch: int=256, upsample_ch: int=256):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList([DecoderLayer(ch=upsample_ch, block_num=block_num)] * len(channels))   # the deepest layer dont have concat block

        self.catblocks = nn.ModuleList([CatBlock(same_ch=in_ch, down_ch=upsample_ch, deepest=True)] + [CatBlock(same_ch=in_ch, down_ch=upsample_ch, deepest=False) for _ in range(len(channels)-1)])

        self.seg_head = SegBlock(in_ch=upsample_ch, middle_ch=128)      # original is channels[0] // 2

    def forward(self, feats: List[torch.Tensor], sz):
        '''
        feats in shape List([1, 256, 100, 100] [1, 512, 50, 50] [1, 1024, 25, 25] [1, 2048, 13, 13])
        '''
        for i, (same_level_feat, catblock, decoder_layer) in enumerate(zip(feats[::-1], self.catblocks, self.decode_layers)):
            if i == 0:  # deepest
                feat = catblock(same_level_feat, None)
            else:
                feat = catblock(same_level_feat, feat)            
                
            feat = decoder_layer(feat)

        feat = self.seg_head(feat, sz)
        return feat

class DecoderLayer(pl.LightningModule):
    '''
    the decoder for each level
    '''
    def __init__(self, ch, block_num=1):
        super(DecoderLayer, self).__init__()

        self.squeeze_channel_block = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),       # normal
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
                )

        self.layer = nn.Sequential(*([DecoderBlock(ch) for _ in range(block_num)] + [self.squeeze_channel_block]))

    def forward(self, feat):
        return self.layer(feat)
    
# class DecoderBlock(pl.LightningModule):
#     '''
#     decoder block in decoder layer
#     '''
#     def __init__(self, ch):
#         super(DecoderBlock, self).__init__()

#         self.block = nn.Sequential(
#                 nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),                  # normal
#                 nn.BatchNorm2d(ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),                  # normal
#                 nn.BatchNorm2d(ch),
#                 )

#     def forward(self, feat):
#         return feat + self.block(feat)

class DecoderBlock(pl.LightningModule):
    '''
    decoder block in decoder layer
    '''
    def __init__(self, ch):
        super(DecoderBlock, self).__init__()

        self.block = ASPP(in_channels=ch, out_channels=ch, atrous_rates=[6,12,18])

    def forward(self, feat):
        return feat + self.block(feat)
    
class CatBlock(pl.LightningModule):
    '''
    block for concat current level feat from middle and below level feat from decoder
    '''
    def __init__(self, same_ch: int=256, down_ch: int=256, out_ch: int=256, deepest=False):
        '''feat_ch: features channels from encoder's same layer'''
        super().__init__()
        
        if deepest:
            ch = same_ch
        else:
            ch = same_ch + down_ch

        self.block = nn.Sequential(
            nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),        # original is out_ch
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
        
        self.upsample_block = nn.ConvTranspose2d(down_ch, down_ch, kernel_size=4, stride=2, padding=1)
        
    def forward(self, same_level_feat, down_feat=None):
        if down_feat is None:
            in_feat = same_level_feat
        else:
            down_feat = self.upsample_block(down_feat)
            down_feat = F.interpolate(down_feat, size=(same_level_feat.size(-2), same_level_feat.size(-1)), mode='bilinear', align_corners=True)
            print(same_level_feat.shape, down_feat.shape)
            in_feat = torch.cat([same_level_feat, down_feat], dim=1)
        return same_level_feat + self.block(in_feat)

class SegBlock(pl.LightningModule):
    '''
    segmenation head
    '''
    def __init__(self, in_ch: int=256, middle_ch: int=128):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, middle_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(middle_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_ch, 2, kernel_size=1, stride=1, padding=0, bias=True)
            )
        
    def forward(self, feat, sz):
        feat = F.interpolate(feat, size=(sz[0], sz[1]), mode='bilinear', align_corners=True)
        feat = self.block(feat)
        return feat

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates: List, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)