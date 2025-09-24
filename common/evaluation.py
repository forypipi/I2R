r""" Evaluation helpers """
from typing import Dict
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """

    # @classmethod
    # def classify_prediction(cls, pred_mask, batch, ignore_index=255):
    #     gt_mask = batch.get('query_mask')

    #     # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
    #     query_ignore_idx = batch.get('query_ignore_idx')
    #     if query_ignore_idx is not None:
    #         assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
    #         query_ignore_idx *= ignore_index
    #         gt_mask = gt_mask + query_ignore_idx
    #         pred_mask[gt_mask == ignore_index] = ignore_index

    #     # compute intersection and union of each episode in a batch
    #     area_inter, area_pred, area_gt = [],  [], []
    #     for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
    #         _pred_mask, _gt_mask = _pred_mask.cpu().float(), _gt_mask.cpu().float()
    #         _inter = _pred_mask[_pred_mask == _gt_mask]

    #         if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
    #             _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
    #         else:
    #             _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
    #         area_inter.append(_area_inter)
    #         area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
    #         area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
    #     area_inter = torch.stack(area_inter).t()
    #     area_pred = torch.stack(area_pred).t()
    #     area_gt = torch.stack(area_gt).t()
    #     area_union = area_pred + area_gt - area_inter

    #     return area_inter, area_union

    # @classmethod
    # def classify_prediction(cls, pred_mask: torch.Tensor, batch: Dict, ignore_index=255):
    #     '''
    #     function is same as original classify_prediction above, work on GPU instead of CPU, and avoid using histc, speed up training
    #     '''
    #     gt_mask = batch['query_mask']

    #     # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
    #     query_ignore_idx = batch['query_ignore_idx']
    #     if query_ignore_idx is not None:
    #         # query_ignore_idx *= ignore_index
    #         # gt_mask = gt_mask + query_ignore_idx
    #         pred_mask[gt_mask == ignore_index] = ignore_index

    #     # compute intersection and union of each episode in a batch
    #     pred_mask, gt_mask = pred_mask.float(), gt_mask.float()
    #     inter = pred_mask.clone()
    #     inter[pred_mask == gt_mask] = 1

    #     area_pred = torch.stack(((pred_mask == 0).sum(dim=(1, 2)), (pred_mask == 1).sum(dim=(1, 2))))
    #     area_gt = torch.stack(((gt_mask == 0).sum(dim=(1, 2)), (gt_mask == 1).sum(dim=(1, 2))))
    #     area_inter = torch.stack((torch.logical_and(pred_mask==0, gt_mask==0).sum(dim=(1, 2)), torch.logical_and(pred_mask==1, gt_mask==1).sum(dim=(1, 2))))

    #     area_union = area_pred + area_gt - area_inter

    #     return area_inter, area_union

    @classmethod
    def classify_prediction(cls, pred_mask: torch.Tensor, target: torch.Tensor, K=2, ignore_index=255):
        # 'K' classes, pred_mask and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.

        assert (pred_mask.dim() in [1, 2, 3])
        assert pred_mask.shape == target.shape
        pred_mask = pred_mask.view(-1)
        target = target.view(-1)
        pred_mask[target == ignore_index] = ignore_index
        intersection = pred_mask[pred_mask == target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
        area_output = torch.histc(pred_mask, bins=K, min=0, max=K-1)
        area_target = torch.histc(target, bins=K, min=0, max=K-1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
