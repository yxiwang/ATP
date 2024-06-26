# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# This implementation is based on:
# https://github.com/lhoyer/DAFormer
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. Licensed under the Apache License, Version 2.0
# A copy of the license is available at resources/license_daformer

import itertools
import logging
import math
import torch


def get_model_base(architecture, backbone):
    if 'daformer_' in architecture and 'proj' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2_proj': '_base_/models/deeplabv2_proj_r50-d8.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/source.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):
    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        if pseudo_random_crop:  # crop deleted
            cfg['_base_'].append(f'_base_/datasets/uda_{source}_to_{target}_{crop}_no_crop.py')
        elif fix_crop:  # crop fixed https://github.com/lhoyer/DAFormer/issues/6
            cfg['_base_'].append(f'_base_/datasets/uda_{source}_to_{target}_{crop}_fix_crop.py')
        else:
            raise FileNotFoundError()
        cfg['_base_'].append(f'_base_/uda/{uda}.py')

        cfg.setdefault('uda', {})
        if method_name in uda and plcrop:
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if method_name in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)

        if method_name in uda:
            cfg['uda']['start_distribution_iter'] = start_distribution_iter
            if pseudo_random_crop:
                cfg['uda']['pseudo_random_crop'] = pseudo_random_crop
                cfg['uda']['cat_max_ratio'] = cat_max_ratio
                crop_size = crop.split('x')
                crop_size = (int(crop_size[0]), int(crop_size[1]))
                cfg['uda']['crop_size'] = crop_size
                cfg['uda']['regen_pseudo'] = regen_pseudo
            cfg['model'].setdefault('auxiliary_head', {})
            cfg['model']['auxiliary_head']['in_channels'] = in_channels
            cfg['model']['auxiliary_head']['in_index'] = contrast_indexes
            cfg['model']['auxiliary_head']['input_transform'] = contrast_mode
            cfg['model']['auxiliary_head']['channels'] = channels
            cfg['model']['auxiliary_head']['num_convs'] = num_convs
            if num_convs == 0:
                if contrast_mode == 'resize_concat':
                    cfg['model']['auxiliary_head']['channels'] = sum(in_channels)
                else:
                    cfg['model']['auxiliary_head']['channels'] = in_channels
            cfg['model']['auxiliary_head'].setdefault('loss_decode', {})
            cfg['model']['auxiliary_head']['loss_decode']['use_reg'] = use_reg
            cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool'] = use_avg_pool
            cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio'] = scale_min_ratio
            cfg['model']['auxiliary_head']['loss_decode']['contrast_temp'] = contrastive_temperature
            cfg['model']['auxiliary_head']['loss_decode']['loss_weight'] = contrastive_weight
            cfg['model']['auxiliary_head']['loss_decode']['reg_relative_weight'] = reg_relative_weight


        if method_name in uda and enable_self_training:
            cfg['uda']['enable_self_training'] = enable_self_training

        # Setup optimizer and schedule
        if method_name in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=1000, metric='mIoU')

        # Construct uda name
        uda_mod = uda

        # Construct config name
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        cfg['name'] = f"{cfg['name_architecture']}_{cfg['name_uda']}_{cfg['name_opt']}_{cfg['name_dataset']}"
        if seed is not None:
            cfg['name'] += f'_seed{seed}'
        cfg['name'] = cfg['name'].replace('.', '.').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    method_name = 'ATP'
    n_gpus = 1
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '640x640'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 1
    rcs_T = None
    plcrop = True
    fix_crop = True  # whether to fix the RandomCrop bug in DAFormer
    start_distribution_iter = 1500
    enable_self_training = True
    pseudo_random_crop = False
    regen_pseudo = False
    cat_max_ratio = 0.75  # used for CBC

    # auxiliary head parameters
    in_channels = 2048  # in_channels = [256, 512, 1024, 2048]
    channels = 512  # default out_dim
    num_convs = 2
    contrast_indexes = 3  # int or list, depending on value of contrast_mode
    contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
    memory_length = 200
    use_reg = False
    use_avg_pool = True
    scale_min_ratio = 0.75  # used for down-sampling
    contrastive_temperature = 100.
    contrastive_weight = 1.0
    reg_relative_weight = 1.0  # reg_weight = reg_relative_weight * loss_weight in auxiliary head

    seeds = [76]  # random seeds

    # dark exclusive
    corresp_root = None
    shift_insensitive_classes = [(0, 5), (8, 11)]
    class_weight_seg = None
    day_ratio = 0.8

    # -------------------------------------------------------------------------
    # GTA -> Cityscapes (ResNet-101)
    # -------------------------------------------------------------------------
    if id == 1:
        # task
        model = ('dlv2_proj', 'r101v1c')
        architecture, backbone = model
        datasets = [
            ('gta', 'cityscapes'),
        ]
        # general
        uda = 'ATP'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        in_channels = 2048
        contrast_indexes = 3  # int or list, depending on value of contrast_mode
        contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
        # reg
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # results
        for seed, mode, (source, target) in itertools.product(seeds, modes, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SYNTHIA -> Cityscapes (ResNet-101)
    # -------------------------------------------------------------------------
    elif id == 2:
        # task
        model = ('dlv2_proj', 'r101v1c')
        architecture, backbone = model
        datasets = [
            ('synthia', 'cityscapes'),
        ]
        # general
        uda = 'ATP'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        in_channels = 2048
        contrast_indexes = 3  # int or list, depending on value of contrast_mode
        contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
        # reg
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # results
        for seed, mode, (source, target) in itertools.product(seeds, modes, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes -> ACDC (ResNet-101)
    # -------------------------------------------------------------------------
    elif id == 3:
        # task
        model = ('dlv2_proj', 'r101v1c')
        architecture, backbone = model
        datasets = [
            ('cityscapes', 'ACDC'),
        ]
        # general
        uda = 'ATP'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        in_channels = 2048
        contrast_indexes = 3  # int or list, depending on value of contrast_mode
        contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
        # reg
        use_reg = True
        reg_relative_weight = 1.0

        # results
        for seed, (source, target) in itertools.product(seeds, datasets):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # GTA -> Cityscapes (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 4:
        # task
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('gta', 'cityscapes')
        ]
        # general
        uda = 'ATP'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # results
        for seed, mode, (source, target) in itertools.product(seeds, modes, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SYNTHIA -> Cityscapes (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 5:
        # task
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('synthia', 'cityscapes')
        ]
        # general
        uda = 'ATP'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # results
        for seed, mode, (source, target) in itertools.product(seeds, modes, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes -> ACDC (MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 6:
        # task
        model = ('daformer_sepaspp_proj', 'mitb5')
        architecture, backbone = model
        datasets = [
            ('cityscapes', 'ACDC')
        ]
        # general
        uda = 'sepico'
        pseudo_random_crop = True
        regen_pseudo = True
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # results
        for seed, mode, (source, target) in itertools.product(seeds, modes, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)

    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
