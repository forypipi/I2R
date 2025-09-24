r""" Extracts intermediate features from given backbone network & layer ids """
import torch
from torch import nn
import torch.nn.functional as F


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None, pool=False, pool_thr=50, support_mask=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            if pool and feat.shape[-1] >= pool_thr:
                    feats.append(F.avg_pool2d(feat.clone(), kernel_size=3, stride=2, padding=1))
            else:
                feats.append(feat.clone())

    prob, img_fore_cls = None, None
    
    if not (support_mask is None):
        feature_size = feat.shape[2:]
        down_mask = F.interpolate(support_mask, size=feature_size, mode='nearest')

        feat = feat * down_mask
        feat = backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = backbone.fc(feat)
        prob = nn.Softmax()(feat)
        img_fore_cls = torch.argmax(prob, dim=1)

    return feats, prob, img_fore_cls


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids, pool=False, pool_thr=50, support_mask=None):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.layer0.forward(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        # original feats return
        # if hid + 1 in feat_ids:
        #     if pool and feat.shape[-1] >= pool_thr:
        #         feats.append(F.avg_pool2d(feat.clone(), kernel_size=3, stride=2, padding=1))
        #     else:
        #         feats.append(feat.clone())

        # return all feats for BAM
        feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
    
    prob, img_fore_cls = None, None
    
    if not (support_mask is None):
        feature_size = feat.shape[2:]
        down_mask = F.interpolate(support_mask, size=feature_size, mode='nearest')

        feat = feat * down_mask
        feat = backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = backbone.fc(feat)
        prob = nn.Softmax()(feat)
        img_fore_cls = torch.argmax(prob, dim=1)

    return feats, prob, img_fore_cls
