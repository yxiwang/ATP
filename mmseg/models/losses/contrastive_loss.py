# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Note that `downscale_label_ratio` method is adapted from: https://github.com/lhoyer/DAFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor >= 1
    if scale_factor == 1:
        return gt.clone()
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # o/w next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def contrast_preparations(feat,
                          mask,
                          use_avg_pool,
                          scale_min_ratio,
                          num_classes,
                          ignore_index):
    # down-sample mask to fit feat
    if use_avg_pool:
        scale_factor = mask.shape[-1] // feat.shape[-1]
        mask = downscale_label_ratio(mask, scale_factor, scale_min_ratio, num_classes, ignore_index).long().detach()
    else:
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode='nearest').long()
    # normalize the feat
    # feat = F.normalize(feat, p=2, dim=1)  # already normalized in proj_head.py
    # transpose the feat shape
    A = feat.size(1)
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, A)
    mask = mask.contiguous().view(-1)

    msk = (mask != ignore_index)
    # remove ignore_index pixels
    mask = mask[msk]
    feat = feat[msk]
    return feat, mask


def proto_reg(feat,
              mean=None,
              contrast_temp=100.,
              contrast_norm=None,
              **kwargs):
    assert mean is not None, 'Parameter `mean` required'
    assert contrast_norm is not None, 'Parameter `contrast_norm` required'
    assert not mean.requires_grad
    assert feat.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    mean_feat = torch.mean(feat, 0, keepdim=True)

    # feat (1, A) x Ave (A, C)
    proto_sim = mean_feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp

    loss = torch.sum(torch.softmax(proto_sim, dim=1).log()) / contrast_norm

    return loss


def proto_contrastive(feat,
                      mask,
                      mean=None,
                      index=-1,
                      contrast_temp=100.,
                      use_avg_pool=True,
                      scale_min_ratio=0.75,
                      num_classes=19,
                      weight=None,
                      class_weight=None,
                      reduction='mean',
                      avg_factor=None,
                      reg_weight=0,
                      ignore_index=255,
                      **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
        feat = feat[index]
        mean = mean[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index)
    assert mean is not None, 'Parameter `mean` required'
    assert not mean.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    # feat (N, A) x Ave (A, C)
    proto_sim = feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp

    # The wrapper function for :func:`F.cross_entropy`
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        proto_sim,
        mask,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = num_classes * np.log(num_classes)
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss


@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    """ContrastiveLoss.

    Args:
        use_reg (bool, optional): Whether to use regularization term.
            Defaults to False.
        use_avg_pool (bool, optional): Whether to use average pooling for down sampling.
            Defaults to True.
        contrast_temp (double, optional): Temperature used in contrastive loss.
            Defaults to 100.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_reg=False,
                 use_avg_pool=True,
                 scale_min_ratio=0.75,
                 num_classes=None,
                 contrast_temp=100.,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 reg_relative_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        assert num_classes is not None
        self.use_reg = use_reg
        self.use_avg_pool = use_avg_pool
        self.scale_min_ratio = scale_min_ratio
        self.contrast_temp = contrast_temp
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_weight = reg_relative_weight
        self.class_weight = get_class_weight(class_weight)

        self.contrast_criterion = proto_contrastive

    def forward(self,
                feat,
                mask,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        # Parameters mean, covariance are sometimes required
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = feat.new_tensor(self.class_weight)
        else:
            class_weight = None
        if isinstance(feat, list):
            if not isinstance(self.loss_weight, list):
                self.loss_weight = [self.loss_weight for _ in range(len(feat))]
            loss_contrast = [self.loss_weight[i] * self.contrast_criterion(
                feat,
                mask,
                weight=weight,
                index=i,
                contrast_temp=self.contrast_temp,
                use_avg_pool=self.use_avg_pool,
                scale_min_ratio=self.scale_min_ratio,
                num_classes=self.num_classes,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                reg_weight=self.reg_weight if self.use_reg else 0,
                **kwargs) for i in range(len(feat))]
            loss_contrast = sum(loss_contrast)
        else:
            loss_contrast = self.loss_weight * self.contrast_criterion(
                feat,
                mask,
                weight=weight,
                contrast_temp=self.contrast_temp,
                use_avg_pool=self.use_avg_pool,
                scale_min_ratio=self.scale_min_ratio,
                num_classes=self.num_classes,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                reg_weight=self.reg_weight if self.use_reg else 0,
                **kwargs)
        return loss_contrast
