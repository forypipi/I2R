from functools import reduce
from operator import add

from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from CAM_Generator.base.feature import extract_feat_res


# class identification module, maps PASCAL/COCO class to ImageNet class
class CIM(pl.LightningModule):
    def __init__(self, later_mask=True):
        '''
        later_mask: True when mask is multiplied after feature extraction(mask size is [bsz*way, H, W]), else should input foreground img as mask args [bsz*way, 3, H, W]
        '''
        super(CIM, self).__init__()

        self.later_mask = later_mask    
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT).cuda()
        self.feat_ids = list(range(4, 17))
        self.extract_feats = extract_feat_res
        nbottlenecks = [3, 4, 6, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

    def cls_head(self, feat):
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.backbone.fc(feat)
        prob = nn.Softmax()(feat)
        return prob
    
    def forward(self, image, mask):
        '''
        return original image prob distribution and only background distribution
        image.shape : [bsz*way, 3, H, W]
        mask.shape : [bsz*way, H, W] if true, else [bsz*way, 3, H, W]
        '''
        self.backbone.eval()
        if self.later_mask:
            with torch.no_grad():
                feat = self.extract_feats(image, self.backbone)
                feature_size = feat.shape[2:]
                down_mask = F.interpolate(mask, size=feature_size, mode='nearest')

                foreground = feat * down_mask

                foreground_cls = self.cls_head(foreground)
                normal_result = self.cls_head(feat)

        else:
            with torch.no_grad():
                foreground_cls = nn.Softmax()(self.backbone(mask))
                normal_result = nn.Softmax()(self.backbone(image))

        return normal_result, foreground_cls

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        cls_score_agg = 0
        support_imgs = batch['support_imgs'].clone()
        support_masks = batch['support_masks'].clone()
        for s_idx in range(nshot):
            batch['support_imgs'] = support_imgs[:, :, s_idx]
            batch['support_masks'] = support_masks[:, :, s_idx]
            shared_masks = self.forward(batch)
            pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
            cls_score_agg += pred_cls.clone()
            logit_mask_agg += logit_seg.clone()
            if nshot == 1:
                return pred_cls, pred_seg

        pred_cls = (cls_score_agg / float(nshot)) >= 0.5
        pred_seg = (logit_mask_agg / float(nshot)).argmax(dim=1)
        return pred_cls, pred_seg

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.parameters(), "lr": self.learning_rate}])
