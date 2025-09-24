import torch
from torch import nn
import torch.nn.functional as F

from .PPM import PPM
import pytorch_lightning as pl


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4

class BaseModel(pl.LightningModule):
    def __init__(self, backbone: pl.LightningModule, args):
        super(BaseModel, self).__init__()

        self.vgg = (args.backbone == "vgg16")
        self.dataset = args.benchmark
        self.args = args
        self.classes = 16 if self.dataset=='pascal' else 61
    
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer0, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
            

        # Base Learner
        self.encoder = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)
        fea_dim = 512 if self.vgg else 2048
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.classes, kernel_size=1))

    def forward(self, x):
        x_size = x.size()

        x = self.encoder(x)
        x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=(x_size[-2], x_size[-1]), mode='bilinear', align_corners=True)

        return x

