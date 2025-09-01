# models/detr_wrapper.py
import torch, torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY, build_model
from detectron2.structures import Instances, Boxes, BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.evaluation import COCOEvaluator

import numpy as np


@META_ARCH_REGISTRY.register()
class DETRWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from torchvision.models.detection import detr_resnet50
        self.model = detr_resnet50(
            weights=cfg.MODEL.WEIGHT, 
            num_classes=cfg.MODEL.NUM_CLASSES, 
            num_queries=cfg.MODEL.EXTRA.NUM_Q)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.training:
            targets = []
            for x in batched_inputs:
                inst: Instances = x["instances"].to(self.device)
                t = {
                    "boxes": inst.gt_boxes.tensor,
                    "labels": inst.gt_classes,
                }
                targets.append(t)
            return self.model(images, targets)   # returns dict of losses

        preds = self.model(images)  # list of dicts: "boxes","scores","labels"
        results = []
        for inp, p in zip(batched_inputs, preds):
            inst = Instances((inp["height"], inp["width"]))
            inst.pred_boxes   = Boxes(p["boxes"])
            inst.scores       = p["scores"]
            inst.pred_classes = p["labels"]
            results.append({"instances": inst})
        return results


def detr_minmax_mapper(dataset_dict):
    """
    Mapper for DETR with no resize/flip and simple min-max normalization [0,1].
    """
    dataset_dict = dataset_dict.copy()

    # Load image as BGR
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # Convert to RGB
    image = image[:, :, ::-1].astype("float32")

    # --- Min-max normalization ---
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:  # avoid division by zero
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)

    # To tensor
    image = torch.as_tensor(image.transpose(2, 0, 1).copy())
    dataset_dict["image"] = image

    # Convert annotations into Instances
    if "annotations" in dataset_dict:
        annos = dataset_dict["annotations"]
        for ann in annos:
            ann["bbox_mode"] = ann.get("bbox_mode", BoxMode.XYWH_ABS)
        instances = utils.annotations_to_instances(annos, image.shape[1:])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    dataset_dict["height"], dataset_dict["width"] = image.shape[1:]
    return dataset_dict



class DETRTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=detr_minmax_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=detr_minmax_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)

        return model