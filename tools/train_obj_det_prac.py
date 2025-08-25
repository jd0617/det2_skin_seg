import os
import time
import argparse
import logging
from datetime import timedelta, datetime
from pathlib import Path

import torch
import random
import numpy as np

from collections import defaultdict
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils import comm

import _init_paths
from dataset.utils import register_dataset, get_records, get_groups_from_records, group_kfold_indices, register_split
from models import get_model
from config import cfg, update_config
from core import DiceScoreEvaluator
from utils.utils import  create_logger


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--cfg',
                    default='/workspace/project/configs/hrnet/w32_ori.yaml',
                    type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser.add_argument('--ds-root', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--record-base', default='', type=str, metavar='PATH',
                    help='path to record base folder (default: none, current dir)')
parser.add_argument('--output-dir', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyTrainer(DefaultTrainer):
    # @classmethod
    # def build_evaluator(cfg):
    #     return DiceScoreEvaluator(cfg, from_logits=cfg.MODEL.FROM_LOGITS, threshold=0.7, mode='train')
    #     # return COCOEvaluator(dataset_name, output_folder or cfg.OUTPUT_DIR)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)

        return model
    

def update_cfg_with_args(cfg, arg_key, arg_value):
    cfg.defrost()

    arg_key = arg_key.upper()

    cfg.arg_key = arg_value

    cfg.freeze()