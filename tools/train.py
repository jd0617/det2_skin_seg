import os
import numpy as np
import argparse


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

def run_nested_cv(base_ds_name: str, cfg)
def main():
    args = parser.parse_args()
    update_config(cfg, args)

    start_datetime = datetime.now()
    start_time = time.monotonic()

