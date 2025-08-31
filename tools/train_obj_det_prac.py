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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm

import _init_paths
from dataset.utils import register_dataset, get_records, get_groups_from_records, group_kfold_indices, register_split
from dataset.utils import register_patch_bin_dataset
from core import VisualizeEval
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

from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
import copy

def keep_gt_mapper(dataset_dict):
    # make a deep copy
    d = copy.deepcopy(dataset_dict)

    # load image
    image = utils.read_image(d["file_name"], format="BGR")
    d["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())

    # DO NOT drop annotations
    if "annotations" in d:
        for ann in d["annotations"]:
            ann["bbox_mode"] = ann.get("bbox_mode", BoxMode.XYWH_ABS)

    d["height"], d["width"] = image.shape[:2]
    return d
  
# def register_coco_binary_remap(name, json_file, img_root):
#     from detectron2.data.datasets import load_coco_json
    
#     def _loader():
#         ds = load_coco_json(json_file, img_root, name)
#         out = []
#         for d in ds:
#             d = d.copy()
#             anns = []
#             for a in d.get('annotations', []):
#                 old = int(a["category_id"])
#                 if old > 1:
#                     a = a.copy()
#                     a["category_id"] = 1
#                 anns.append(a)
#             d["annotations"] = anns
#             out.append(d)
#         return out
    
#     DatasetCatalog.register(name, _loader)



def update_cfg_with_args(cfg, arg_key, arg_value):
    cfg.defrost()

    arg_key = arg_key.upper()

    cfg.arg_key = arg_value

    cfg.freeze()


def main():
    args = parser.parse_args()
    if not hasattr(comm, "REFERENCE_WORLD_SIZE"):
        comm.REFERENCE_WORLD_SIZE = comm.get_world_size()

    start_datetime = datetime.now()
    start_time = time.monotonic()
    
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
        # 'COCO-Detection/retinanet_R_50_FPN_1x.yaml'

    ))

    update_config(cfg, args)
    
    # update_cfg_with_args(cfg, 'OUTPUT_DIR', args.output_dir)

    final_output_dir = Path(os.path.join(cfg.RECORD_BASE, cfg.OUTPUT_DIR))
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_train.log'.format(time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # keep = ["level_0", "level_1", "level_2"]

    # register_coco_instances(cfg.DATASETS.TRAIN[0], {}, cfg.DATASETS.TRAIN_ANNO_DIR, cfg.DATASETS.IMG_DIR)
    # register_coco_instances(cfg.DATASETS.TEST[0], {}, cfg.DATASETS.TEST_ANNO_DIR, cfg.DATASETS.IMG_DIR)

    # register_coco_binary_remap(cfg.DATASETS.TRAIN[0], cfg.DATASETS.TRAIN_ANNO_DIR, cfg.DATASETS.IMG_DIR)
    # register_coco_binary_remap(cfg.DATASETS.TEST[0], cfg.DATASETS.TRAIN_ANNO_DIR, cfg.DATASETS.IMG_DIR)

    register_patch_bin_dataset(
        cfg.DATASETS.TRAIN[0],
        json_file=cfg.DATASETS.TRAIN_ANNO_DIR,
        img_root=cfg.DATASETS.IMG_DIR,
        extra_key=["patient_id"]
    )

    register_patch_bin_dataset(
        cfg.DATASETS.TEST[0],
        json_file=cfg.DATASETS.TEST_ANNO_DIR,
        img_root=cfg.DATASETS.IMG_DIR,
        extra_key=["patient_id"]
    )

    set_seed(cfg.SEED)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    logger.info("=====> Testing <=====")

    evaluator = DatasetEvaluators([
                COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=final_output_dir),
                VisualizeEval(cfg.DATASETS.TEST[0], output_dir=final_output_dir,
                              max_images=100, score_thresh=0.05, topk=300)
            ])
    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=keep_gt_mapper)
    results = inference_on_dataset(trainer.model, test_loader, evaluator) 

    print(results)

    end_time = time.monotonic()
    end_datetime = datetime.now()

    duration = timedelta(seconds=end_time - start_time)
    logger.info("Experiment start at: {}".format(start_datetime))
    logger.info("Experiment End Time: {}".format(end_datetime))
    logger.info("Time taken: {}".format(duration))
    logger.info("Results available at: {}".format(final_output_dir))

    
if __name__ == "__main__":
    main()


