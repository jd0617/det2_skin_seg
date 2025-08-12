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
    
def build_inner_cfg(
        cfg_file, train_name, test_name, out_dir, num_classes,
        batch_size, 

)

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

            ifold_cfg = 




    
def main():
    args = parser.parse_args()
    update_config(cfg, args)

    start_datetime = datetime.now()
    start_time = time.monotonic()

def run_nested_cv(base_dataset_name: str,
                  cfg_yaml_path: str,
                  num_classes: int,
                  k_outer: int = 5,
                  k_inner: int = 3,
                  hp_grid = None,
                  hrnet_name: str = "hrnet_w32",
                  mask_on: bool = True):




    for o_fold, (outer_tr_idx, outer_te_idx) in enumerate(group_kfold_indices(groups, n_splits=k_outer, seed=42)):
        # INNER SEARCH on outer-train
        hp_scores = defaultdict(list)
        for i_fold, (inner_tr_rel, inner_va_rel) in enumerate(group_kfold_indices(groups[outer_tr_idx], n_splits=k_inner, seed=123)):
            inner_tr_idx = outer_tr_idx[inner_tr_rel]
            inner_va_idx = outer_tr_idx[inner_va_rel]

            inner_tr_name = f"o{o_fold}_inner_tr_{i_fold}"
            inner_va_name = f"o{o_fold}_inner_va_{i_fold}"
            register_split(inner_tr_name, records, inner_tr_idx, base_dataset_name)
            register_split(inner_va_name, records, inner_va_idx, base_dataset_name)

            for hp_id, hp in enumerate(hp_grid):
                outdir = f"./outs/inner/o{o_fold}/fold{i_fold}/hp{hp_id}"
                cfg = build_cfg(cfg_yaml_path, inner_tr_name, inner_va_name, outdir,
                                num_classes=num_classes,
                                ims_per_batch=4,
                                base_lr=hp["base_lr"], max_iter=hp["max_iter"],
                                hrnet_name=hrnet_name, mask_on=mask_on)
                trainer = MyTrainer(cfg)
                trainer.resume_or_load(False)
                trainer.train()

                evaluator = COCOEvaluator(inner_va_name, output_dir=cfg.OUTPUT_DIR)
                val_loader = build_detection_test_loader(cfg, inner_va_name)
                res = inference_on_dataset(trainer.model, val_loader, evaluator)
                key = "segm" if mask_on and "segm" in res else "bbox"
                ap = float(res[key]["AP"])
                hp_scores[hp_id].append(ap)

        # pick best HP by mean AP across inner folds
        best_hp_id = max(hp_scores, key=lambda k: np.mean(hp_scores[k]))
        best_hp = hp_grid[best_hp_id]
        best_mean = float(np.mean(hp_scores[best_hp_id]))

        # register outer train/test
        outer_tr_name = f"outer_tr_{o_fold}"
        outer_te_name = f"outer_te_{o_fold}"
        register_split(outer_tr_name, records, outer_tr_idx, base_dataset_name)
        register_split(outer_te_name, records, outer_te_idx, base_dataset_name)

        # train final model on full outer-train with best HPs, eval on outer-test
        outdir = f"./outs/outer/fold{o_fold}"
        cfg = build_cfg(cfg_yaml_path, outer_tr_name, outer_te_name, outdir,
                        num_classes=num_classes,
                        ims_per_batch=4,
                        base_lr=best_hp["base_lr"], max_iter=best_hp["max_iter"],
                        hrnet_name=hrnet_name, mask_on=mask_on)
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(False)
        trainer.train()

        evaluator = COCOEvaluator(outer_te_name, output_dir=cfg.OUTPUT_DIR)
        test_loader = build_detection_test_loader(cfg, outer_te_name)
        res = inference_on_dataset(trainer.model, test_loader, evaluator)
        key = "segm" if mask_on and "segm" in res else "bbox"

        outer_results.append({
            "fold": o_fold,
            "best_hp": best_hp,
            "best_inner_mean_ap": best_mean,
            "test_metrics": res[key]
        })

    return outer_results