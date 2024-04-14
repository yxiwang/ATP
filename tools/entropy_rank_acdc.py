# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

import argparse
import os
import shutil

import mmcv
import torch
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import numpy as np


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg

def prob_2_entropy(prob):
    n, c, h, w = prob.size()
    entropy = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    return entropy

def cluster_subdomain(entropy_list, lambda1):
    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]
       
    easy_split = entropy_rank[: int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank) * lambda1) :]

    for item in easy_split:
        print("item", item[0])
        folder_path, file_folder = item[0].split('train/')[0], item[0].split('train/')[1]
        file_set, file_name = file_folder.split('/')[0],file_folder.split('/')[1]
        target_path = os.path.join(folder_path, "easy", file_set)
        os.makedirs(target_path, exist_ok=True)
        target_file_path = os.path.join(target_path, file_name)
        shutil.copy2(item[0], target_file_path)    
    print("finised easy")  

    for item in hard_split:
        folder_path, file_folder = item[0].split('train/')[0], item[0].split('train/')[1]
        file_set, file_name = file_folder.split('/')[0],file_folder.split('/')[1]
        target_path = os.path.join(folder_path, "hard", file_set)
        os.makedirs(target_path, exist_ok=True)
        target_file_path = os.path.join(target_path, file_name)
        shutil.copy2(item[0], target_file_path)     
    print("finised hard")

    with open('easy_split.txt', 'w+') as f:
        for item in easy_split:
            f.write('%s\n' % item)

    with open('hard_split.txt', 'w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)
    return copy_list

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--palette',
        action='store_true',
        help='Whether to use palette in format.'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True 

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print('dataset', cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    #print('dataset', dataset.ann_dir)
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    #print(args.checkpoint)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    print('successfully load checkpoints')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    entropy_list = []
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)
    #print(data_loader['img_metas'])

    entropy = []
    for batch in data_loader:
        img = batch['img']
        #print("img", img[0].shape)
        img_metas = batch["img_metas"]
        #print("img_metas", img_metas)
        file_name = [img_metas[0].data[0][0]["filename"] for i in range(len(img_metas))]
        #print("name", file_name[0])
        outputs = model.encode_decode(img[0], img_metas)

        entropy = prob_2_entropy(F.softmax(outputs, dim=1))
        #print("entropy", entropy.mean().item())
        entropy_list.append((file_name, entropy.mean().item()))

    """
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
        #entropy = prob_2_entropy(F.softmax(outputs))
        #entropy_list.append(dataset['img_meta']['name'], entropy.mean().item())
        rank, items = get_dist_info()
        print("rank", rank)
        print("items", items)
        
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

        rank, items = get_dist_info()
        
        entropy = prob_2_entropy(F.softmax(outputs, dim=1))
        #entropy_list.append(dataset['img_meta']['name'], entropy.mean().item())
    print(entropy.shape)
    """
    cluster_subdomain(entropy_list, lambda1=0.3)



if __name__ == '__main__':
    main()
