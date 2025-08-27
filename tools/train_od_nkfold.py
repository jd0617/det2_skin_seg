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
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm

import _init_paths
from dataset.utils import register_dataset, get_records, get_groups_from_records, group_kfold_indices, register_split
from models import get_model
from config import cfg, update_config
from core import DiceScoreEvaluator
from utils.utils import  create_logger


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--cfg',
                    default='/workspace/project/configs/hrnet/w32_obj_det.yaml',
                    type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser.add_argument('--ds-root', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--record-base', default='', type=str, metavar='PATH',
                    help='path to record base folder (default: none, current dir)')
parser.add_argument('--output-dir', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
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
    
def register_coco_binary_remap(name, json_file, img_root):
    from detectron2.data.datasets import load_coco_json
    
    def _loader():
        ds = load_coco_json(json_file, img_root, name)
        out = []
        for d in ds:
            d = d.copy()
            anns = []
            for a in d.get('annotations', []):
                old = int(a["category_id"])
                if old > 1:
                    a = a.copy()
                    a["category_id"] = 1
                anns.append(a)
            d["annotations"] = anns
            out.append(d)
        return out
    
    DatasetCatalog.register(name, _loader)

def build_cfg(cfg_file, train_name, test_name, output_dir, num_samples=0):

    # cfg = cfg.clone()
    
    # if isinstance(cfg_file, str):
    #     cfg.merge_from_file(cfg)
    # else:
    #     cfg.merge_from_other_cfg(cfg_file)

    train_name = str(train_name)
    test_name = str(test_name)
    output_dir = str(output_dir)

    cfg = cfg_file.clone()
    cfg.defrost()

    batch_size = cfg.SOLVER.IMS_PER_BATCH
    epochs = cfg.SOLVER.EPOCHS
    total_steps = (num_samples // batch_size) * epochs

    cfg.DATASETS.TRAIN = train_name
    cfg.DATASETS.TEST = test_name
    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.max_iter = total_steps

    cfg.freeze()

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Total training steps: {total_steps}")

    return cfg

def update_cfg_with_args(cfg, arg_key, arg_value):
    cfg.defrost()

    arg_key = arg_key.upper()

    cfg.arg_key = arg_value

    cfg.freeze()

def run_nested_cv(logger, base_ds_name, cfg, output_dir, k_outer=5, k_inner=3, seed=24):
    records = get_records(base_ds_name)
    groups = get_groups_from_records(records)
    idx_all = np.arange(len(records))

    outer_results = []

    outer_split = group_kfold_indices(groups, n_splits=k_outer)

    best_score = -1
    best_model_path = None

    for o_fold, (outer_tr_idx, outer_te_idx) in enumerate(outer_split):
        
        ofold_prefix = f"ofold_{o_fold}"
        ofold_output_dir = Path(output_dir) / ofold_prefix
        ofold_output_dir.mkdir(parents=True, exist_ok=True)

        test_name = f"o{o_fold}_te"

        logger.info(f"Running outer fold {o_fold}")

        