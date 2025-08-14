import os
import time
import numpy as np
import argparse
import logging
from datetime import timedelta, datetime
from pathlib import Path

from collections import defaultdict
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo

import _init_paths
from dataset.utils import register_dataset, get_records, get_groups_from_records, group_kfold_indices, register_split
from models import get_model
from config import cfg, update_config


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--cfg',
                    default='/home/jd0617/Projects/Skin_Patch/skin_patch_segmentation/exps/hrnet/w32_only_complt.yaml',
                    type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser.add_argument('--ds-root', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--record-base', default='', type=str, metavar='PATH',
                    help='path to record base folder (default: none, current dir)')
parser.add_argument('--output-dir', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, output_folder or cfg.OUTPUT_DIR)
    
def bulid_trainer()
    
def build_inner_cfg(cfg_file, train_name, test_name, output_dir):

    cfg = get_cfg()
    
    if isinstance(cfg_file, str):
        cfg.merge_from_file(cfg)
    else:
        cfg.merge_from_other_cfg(cfg_file)

    cfg.TRAIN_NAME = train_name
    cfg.TEST_NAME = test_name
    cfg.OUTPUT_DIR = output_dir

    return cfg


def run_nested_cv(base_ds_name: str, cfg, output_dir, num_classes:int, 
                  k_outer:int=5, k_inner:int=3,
                  mask_on: bool = True, seed:int=24):

    records = get_records(base_ds_name)
    groups = get_groups_from_records(records)
    idx_all = np.arange(len(records))

    outer_results = []

    for o_fold, (outer_tr_idx, outer_te_idx) in enumerate(group_kfold_indices(groups, n_splits=k_outer, seed=seed)):
        
        hp_scores = defaultdict(list)

        ofold_prefix = f"ofold_{o_fold}"
        ofold_output_dir = Path(output_dir) / ofold_prefix

        perf_grid = {}

        model_grid = {}

        for i_fold, (inner_tr_rel, inner_va_rel) in enumerate(
            group_kfold_indices(groups[outer_tr_idx], n_splits=k_inner, seed=123)):
            
            ifold_prefix = f"ifold_{i_fold}"
            ifold_output_dir = Path(ofold_output_dir) / ifold_prefix

            inner_tr_idx = outer_tr_idx[inner_tr_rel]
            inner_va_idx = outer_tr_idx[inner_va_rel]

            inner_tr_name = f"o{o_fold}_inner_tr_{i_fold}"
            inner_va_name = f"o{o_fold}_inner_va_{i_fold}"

            register_split(inner_tr_name, records, inner_tr_idx, base_ds_name)
            register_split(inner_va_name, records, inner_va_idx, base_ds_name)

            ifold_cfg = build_inner_cfg(cfg, inner_tr_name, inner_va_name, ifold_output_dir)

            trainer = MyTrainer(ifold_cfg)
            trainer.resume_or_load(False)
            trainer.train()

            evaluator = COCOEvaluator(inner_va_name, output_dir=ifold_output_dir)
            val_loader = build_detection_test_loader(ifold_cfg, inner_va_name)

            val_res = inference_on_dataset(trainer.model, val_loader, evaluator)
            key = "segm" if mask_on and "segm" in val_res else "bbox"

            model_grid[]










    
def main():
    args = parser.parse_args()
    update_config(cfg, args)

    start_datetime = datetime.now()
    start_time = time.monotonic()
