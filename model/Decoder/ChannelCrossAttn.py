from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from einops import rearrange, repeat


class ChannelCrossAttn(pl.LightningModule):

    def __init__(self, dim=256, squeeze_rate=4):
        super().__init__()

        self.CCA = nn.Sequential(
            nn.Linear(dim, dim // squeeze_rate),
            nn.ReLU(),
            nn.Linear(dim // squeeze_rate, dim // 2),
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x and y in shape of [b, 256, 30, 30], generate x channel weight
        x_maxpool = F.adaptive_avg_pool2d(x, 1)
        y_maxpool = F.adaptive_avg_pool2d(y, 1)

        x_mix_maxpool = torch.cat([x_maxpool, y_maxpool], 1)   # [bsz, 512, 1, 1]
        x_mix_maxpool = rearrange(x_mix_maxpool, 'b c h w -> b (c h w)')
        x_maxpool_weight = self.CCA(x_mix_maxpool)

        x_avgpool = F.adaptive_max_pool2d(x, 1)
        y_avgpool = F.adaptive_max_pool2d(y, 1)
        x_mix_avgpool = torch.cat([x_avgpool, y_avgpool], 1)   # [bsz, 512, 1, 1]
        x_mix_avgpool = rearrange(x_mix_avgpool, 'b c h w -> b (c h w)')
        x_avgpool_weight = self.CCA(x_mix_avgpool)

        x_channel_weight: torch.Tensor = repeat(self.sigmoid(x_maxpool_weight + x_avgpool_weight), 'b c -> b c h w', h=x.size(-2), w=x.size(-1))

        return x_channel_weight