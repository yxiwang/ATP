# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.utils.ours_transforms import RandomCrop, RandomCropNoProd

from mmseg.models.utils.proto_estimator import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class SePiCo(UDADecorator):

    def __init__(self, **cfg):
        super(SePiCo, self).__init__(**cfg)
        # basic setup
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']

        # for ssl
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        assert self.mix == 'class'
        self.enable_self_training = cfg['enable_self_training']
        self.enable_strong_aug = cfg['enable_strong_aug']
        self.push_off_self_training = cfg.get('push_off_self_training', False)

        # configs for contrastive
        self.proj_dim = cfg['model']['auxiliary_head']['channels']
        self.contrast_mode = cfg['model']['auxiliary_head']['input_transform']
        self.calc_layers = cfg['model']['auxiliary_head']['in_index']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.enable_avg_pool = cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool']
        self.scale_min_ratio = cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio']

        # iter to start cl
        self.start_distribution_iter = cfg['start_distribution_iter']

        # for prod strategy (CBC)
        self.pseudo_random_crop = cfg.get('pseudo_random_crop', False)
        self.crop_size = cfg.get('crop_size', (640, 640))
        self.cat_max_ratio = cfg.get('cat_max_ratio', 0.75)
        self.regen_pseudo = cfg.get('regen_pseudo', False)
        self.prod = cfg.get('prod', True)

        # feature storage for contrastive
        self.feat_distributions = None
        self.ignore_index = 255

        # BankCL memory length
        self.memory_length = cfg.get('memory_length', 0)  # 0 means no memory bank

        # init distribution
        if self.contrast_mode == 'multiple_select':
            self.feat_distributions = {}
            for idx in range(len(self.calc_layers)):
                self.feat_distributions[idx] = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                              memory_length=self.memory_length)
        else:  # 'resize_concat' or None
            self.feat_distributions = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                     memory_length=self.memory_length)

        # ema model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def random_crop(self, image, gt_seg, prod=True):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        assert self.pseudo_random_crop
        image = image.permute(0, 2, 3, 1).contiguous()
        gt_seg = gt_seg
        res_img, res_gt = [], []
        for img, gt in zip(image, gt_seg):
            results = {'img': img, 'gt_semantic_seg': gt, 'seg_fields': ['gt_semantic_seg']}
            results = RC(results)
            img, gt = results['img'], results['gt_semantic_seg']
            res_img.append(img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        return image, gt_seg

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, target_gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # weak_img equals to the easy split images, and the weak_target_img equals to the hard split images 
        weak_easy_img, weak_hard_img = img.clone(), target_img.clone()

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # Generate pseudo-label for easy split
        ema_easy_logits = self.get_ema_model().encode_decode(weak_easy_img, img_metas)
        ema_easy_softmax = torch.softmax(ema_easy_logits.detach(), dim=1)
        _, pseudo_label_easy = torch.max(ema_easy_softmax, dim=1)
        pseudo_label_easy = pseudo_label_easy.unsqueeze(1)

        # Generate pseudo-label
        ema_hard_logits = self.get_ema_model().encode_decode(weak_hard_img, target_img_metas)
        ema_hard_softmax = torch.softmax(ema_hard_logits.detach(), dim=1)
        pseudo_prob_hard, pseudo_label_hard = torch.max(ema_hard_softmax, dim=1)
        ps_large_p = pseudo_prob_hard.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label_hard.cpu()))
        pseudo_weight_hard = torch.sum(ps_large_p).item() / ps_size

        # pseudo RandomCrop
        if self.pseudo_random_crop:
            weak_hard_img, pseudo_label_hard = self.random_crop(weak_hard_img, pseudo_label_hard, prod=self.prod)
            if self.regen_pseudo:
                # Re-Generate pseudo-label
                ema_hard_logits = self.get_ema_model().encode_decode(weak_hard_img, target_img_metas)
                ema_hard_softmax = torch.softmax(ema_hard_logits.detach(), dim=1)
                pseudo_prob_hard, pseudo_label_hard = torch.max(ema_hard_softmax, dim=1)
                ps_large_p = pseudo_prob_hard.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label_hard.cpu()))
                pseudo_weight_hard = torch.sum(ps_large_p).item() / ps_size
            hard_img = weak_hard_img.clone()

        if self.enable_strong_aug:
            easy_img, pseudo_label_easy = strong_transform(
                strong_parameters,
                data=weak_easy_img,
                target=pseudo_label_easy
            )
            hard_img, _ = strong_transform(
                strong_parameters,
                data=hard_img,
                target=pseudo_label_hard.unsqueeze(1)
            )

        pseudo_weight_hard = pseudo_weight_hard * torch.ones(
            pseudo_label_hard.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight_hard[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight_hard[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight_hard.shape, device=dev)

        pseudo_label_easy = pseudo_label_easy.clone().detach()

        # update distribution
        ema_easy_feat = self.get_ema_model().extract_auxiliary_feat(weak_easy_img)
        mean = {}
        covariance = {}
        bank = {}
        if self.contrast_mode == 'multiple_select':
            for idx in range(len(self.calc_layers)):
                feat, mask = contrast_preparations(ema_easy_feat[idx], pseudo_label_easy, self.enable_avg_pool,
                                                   self.scale_min_ratio, self.num_classes, self.ignore_index)
                self.feat_distributions[idx].update_proto(features=feat.detach(), labels=mask)
                mean[idx] = self.feat_distributions[idx].Ave
                covariance[idx] = self.feat_distributions[idx].CoVariance
                bank[idx] = self.feat_distributions[idx].MemoryBank
        else:  # 'resize_concat' or None
            feat, mask = contrast_preparations(ema_easy_feat, pseudo_label_easy, self.enable_avg_pool,
                                               self.scale_min_ratio, self.num_classes, self.ignore_index)
            self.feat_distributions.update_proto(features=feat.detach(), labels=mask)
            mean = self.feat_distributions.Ave
            covariance = self.feat_distributions.CoVariance
            bank = self.feat_distributions.MemoryBank

        # source ce + cl
        easy_mode = 'dec'  # stands for ce only
        easy_losses = self.get_model().forward_train(easy_img, img_metas, pseudo_label_easy, return_feat=False,
                                                       mean=mean, covariance=covariance, bank=bank, mode=easy_mode)
        easy_loss, source_log_vars = self._parse_losses(easy_losses)
        log_vars.update(add_prefix(source_log_vars, 'src'))
        easy_loss.backward()

        if self.local_iter >= self.start_distribution_iter:
            # target cl
            pseudo_lbl = pseudo_label_hard.clone()  # pseudo label should not be overwritten
            pseudo_lbl[pseudo_weight_hard == 0.] = self.ignore_index
            pseudo_lbl = pseudo_lbl.unsqueeze(1)
            hard_losses = self.get_model().forward_train(hard_img, target_img_metas, pseudo_lbl, return_feat=False,
                                                           mean=mean, covariance=covariance, bank=bank, mode='aux')
            hard_loss, target_log_vars = self._parse_losses(hard_losses)
            log_vars.update(add_prefix(target_log_vars, 'tgt'))
            hard_loss.backward()

        local_enable_self_training = \
            self.enable_self_training and \
            (not self.push_off_self_training or self.local_iter >= self.start_distribution_iter)

        # mixed ce (ssl)
        if local_enable_self_training:
            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(pseudo_label_easy)

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((easy_img[i], hard_img[i])),
                    target=torch.stack((pseudo_label_easy[i].squeeze(), pseudo_label_hard[i])))
                _, pseudo_weight_hard[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight_hard[i])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            # Train on mixed images
            mix_losses = self.get_model().forward_train(mixed_img, img_metas, mixed_lbl, pseudo_weight_hard,
                                                        return_feat=False, mode='dec')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(add_prefix(mix_log_vars, 'mix'))
            mix_loss.backward()

        self.local_iter += 1

        return log_vars

