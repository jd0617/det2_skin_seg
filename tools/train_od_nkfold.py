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
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm

import _init_paths
from dataset.utils import register_dataset, get_records, get_groups_from_records, group_kfold_indices, register_split
from dataset.utils import register_patch_bin_dataset
from config import cfg, update_config
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
    cfg.DATASETS.TEST = (test_name, )
    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.MAX_ITER = total_steps
    cfg.SOLVER.STEPS = (int(total_steps*0.8), int(total_steps*0.95))

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
    records = DatasetCatalog.get(base_ds_name)
    groups = get_groups_from_records(records)
    idx_all = np.arange(len(records))

    outer_results = {}

    outer_split = group_kfold_indices(groups, n_splits=k_outer)

    best_score = -1
    best_model_path = None

    for o_fold, (outer_tr_idx, outer_te_idx) in enumerate(outer_split):
        
        ofold_prefix = f"ofold_{o_fold}"
        ofold_output_dir = Path(output_dir) / ofold_prefix
        ofold_output_dir.mkdir(parents=True, exist_ok=True)

        test_name = f"o{o_fold}_te"

        logger.info(f"Running outer fold {o_fold}")

        ofold_cfg = build_cfg(cfg, f"o{o_fold}_tr", test_name, ofold_output_dir)

        inner_split = group_kfold_indices(groups[outer_tr_idx], n_splits=k_inner)

        for i_fold, (inner_tr_rel, inner_va_rel) in enumerate(inner_split):
            
            ifold_prefix = f"ifold_{i_fold}"
            ifold_output_dir = Path(ofold_output_dir) / ifold_prefix
            ifold_output_dir.mkdir(parents=True, exist_ok=True)

            inner_tr_idx = outer_tr_idx[inner_tr_rel]
            inner_va_idx = outer_tr_idx[inner_va_rel]

            inner_tr_name = f"o{o_fold}_inner_tr_{i_fold}"
            inner_va_name = f"o{o_fold}_inner_va_{i_fold}"

            register_split(inner_tr_name, records, inner_tr_idx, base_name_with_classes=base_ds_name)
            register_split(inner_va_name, records, inner_va_idx, base_name_with_classes=base_ds_name)

            ifold_cfg = build_cfg(cfg, inner_tr_name, inner_va_name, ifold_output_dir, len(inner_tr_rel))

            logger.info(f"====> Running inner fold {o_fold}-{i_fold}")
            logger.info(f"===> Training steps: {ifold_cfg.SOLVER.MAX_ITER}")

            trainer = MyTrainer(ifold_cfg)
            trainer.resume_or_load(False)
            trainer.train()

            evaluator = COCOEvaluator(inner_va_name, output_dir=ifold_output_dir)
            val_loader = build_detection_test_loader(ifold_cfg, inner_va_name)
            results = inference_on_dataset(trainer.model, val_loader, evaluator)

            ap = results['bbox']['AP']

            if ap > best_score:
                best_score = ap
                best_model_path = os.path.join(ifold_output_dir, "model_final.pth")

        logger.info(f"Testing outer fold {o_fold}")

        register_split(test_name, records, outer_te_idx, base_name_with_classes=base_ds_name)

        ofold_cfg = build_cfg(cfg, "", test_name, ofold_output_dir)
        update_cfg_with_args(ofold_cfg, 'MODEL.WEIGHTS', best_model_path)
        # ofold_cfg.MODEL.WEIGHTS = best_model_path
        model = build_model(ofold_cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()
        
        test_evaluator = COCOEvaluator(test_name, output_dir=ofold_output_dir)
        test_loader = build_detection_test_loader(cfg, test_name)
        test_results = inference_on_dataset(model, test_loader, test_evaluator)

        outer_results[o_fold] = test_results

    return outer_results


def main():
    args = parser.parse_args()
    if not hasattr(comm, "REFERENCE_WORLD_SIZE"):
        comm.REFERENCE_WORLD_SIZE = comm.get_world_size()

    start_datetime = datetime.now()
    start_time = time.monotonic()

    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    ))

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    update_config(cfg, args)

    # DATASET_NAME = "all_ds"

    register_patch_bin_dataset(
        cfg.DATASETS.DATASET,
        json_file=cfg.DATASETS.ANNO_DIR,
        img_root=cfg.DATASETS.IMG_DIR,
        extra_key=["patient_id"]
    )

    ncv_result = run_nested_cv(logger, base_ds_name=cfg.DATASETS.DATASET, cfg=cfg, output_dir=cfg.OUTPUT_DIR,
                  k_outer=cfg.K_FOLD, k_inner=cfg.VAL_K_FOLD, seed=cfg.SEED)
    
    for k, v in ncv_result.items():
        logger.info(f"Config: {k}")
        logger.info(f"Result: {v}")
        spacer = '-' * 100
        logger.info(spacer)

    end_time = time.monotonic()
    end_datetime = datetime.now()

    duration = timedelta(seconds=end_time - start_time)
    logger.info("Experiment start at: {}".format(start_datetime))
    logger.info("Experiment End Time: {}".format(end_datetime))
    logger.info("Time taken: {}".format(duration))
    logger.info("Results available at: {}".format(final_output_dir))


if __name__ == "__main__":
    main()
