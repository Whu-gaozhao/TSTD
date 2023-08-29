# class Base:
#     def __init__(self,a,b=1):
#         print("a:",a)
#         print("b:",b)

# class A(Base):
#     def __init__(self,c,d=1,**kwargs):
#         super().__init__(**kwargs)
#         print("c:",c)
#         print("d:",d)

# a=A('c',a=2)

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    return args


args = parse_args()

cfg = Config.fromfile(args.config)
cfg.work_dir = osp.join('./work_dirs',
                        osp.splitext(osp.basename(args.config))[0])
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
logger_name = 'mmseg'

logger = get_root_logger(
    log_file=log_file, log_level=cfg.log_level, name=logger_name)

model = build_model(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

logger.info(f'Model:\n{model}')
