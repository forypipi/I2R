from .grad_cam import GradCAM

import torch
from torch.nn import functional as F

from util.util import BNC2BCHW


class XGradCAM(GradCAM):
    def get_mask(self, img: torch.Tensor,
                 target_class: torch.Tensor,
                 target_layer: str):

        B, C, H, W = img.size()
        self.model.eval()
        self.model.zero_grad()

        # class-specific backpropagation
        logits = self.model(img)
        target = self._encode_one_hot(target_class, logits)
        logits.backward(gradient=target, retain_graph=True)

        # get feature maps and gradients
        feature_maps = self._find(self.feature_maps, target_layer)
        gradients = self._find(self.gradients, target_layer)


        # In a typical CNN, self.feature_maps and gradients have shape [B, C, H, W].
        # For ViT (Vision Transformer), feature_maps and gradients are usually [B, N, C],
        # where N is the number of tokens (patches), and C is the embedding dimension.

        # To adapt to ViT, you need to reshape or permute the tensors accordingly.
        # For example, if feature_maps is [B, N, C], you may want to treat N as "spatial" dimension.
        # You can transpose to [B, C, N] and treat N similar to H*W in CNNs.

        # Example for ViT:
        if feature_maps.dim() == 3:  # [B, N, C]
            feature_maps = BNC2BCHW(feature_maps[:,1:,:])
            gradients = BNC2BCHW(gradients[:,1:,:])

        # generate CAM
        with torch.no_grad():
            sum_feature_maps = torch.sum(feature_maps, dim=(2, 3))
            sum_feature_maps = sum_feature_maps[:, :, None, None]
            eps = 1e-7
            weights = gradients * feature_maps / (sum_feature_maps + eps)
            weights = torch.sum(weights, dim=(2, 3))
            weights = weights[:, :, None, None]
            cam = torch.mul(feature_maps, weights).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)

        return cam