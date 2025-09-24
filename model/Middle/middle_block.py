from typing import List
import torch
import torch.nn as nn
import pytorch_lightning as pl


class Middle(pl.LightningModule):
    '''
    process between encoder and decoder
    '''
    def __init__(self, feat_ids: List, channels: List, expand_feat: int = 2, out_ch: int=256):
        super(Middle, self).__init__()

        self.layers = nn.ModuleList()

        for feat_channel, sim_channel in zip(channels, feat_ids):
            in_ch = expand_feat * feat_channel + sim_channel
            layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(p=0.5)
                )
            self.layers.append(layer)

    def forward(self, feats: List) -> List[torch.Tensor]:
        '''
        transform encoder output to decoder input
        '''

        outputs = []

        for feat, layer in zip(feats, self.layers):
            outputs.append(layer(feat))

        return outputs