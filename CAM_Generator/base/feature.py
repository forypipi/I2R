r""" Extracts intermediate features from given backbone network & layer ids """
import torch.nn.functional as F
from torchvision.models import resnet50

def extract_feat_res(img, backbone: resnet50):
    r""" Extract intermediate features from ResNet"""

    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    feat = backbone.layer1(feat)
    feat = backbone.layer2(feat)
    feat = backbone.layer3(feat)
    feat = backbone.layer4(feat)

    return feat
