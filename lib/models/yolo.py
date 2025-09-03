import torch, torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY, build_model
from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.evaluation import COCOEvaluator

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss



from .utils import minmax_mapper

@META_ARCH_REGISTRY.register()
class UltralyticsYOLO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # load YOLOv8 model from Ultralytics
        # e.g. cfg.MODEL.YOLO.MODEL_NAME = "yolov8n.pt"
        yolo = YOLO(cfg.MODEL.EXTRA.MODEL_NAME)
        self.backbone = yolo.model
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone.to(self.device)

        self.score_thresh = cfg.MODEL.EXTRA.SCORE_THRESH_TEST
        self.max_det      = cfg.TEST.DETECTIONS_PER_IMAGE

        self.criterion = v8DetectionLoss(self.backbone)

    def forward(self, batched_inputs):

        device = next(self.parameters()).device

        images = torch.stack([xe['image'].to(device, non_blocking=True) for xe in batched_inputs])

        if self.training:
            # convert Detectron2 Instances -> YOLO labels
            labels = []
            for x in batched_inputs:
                inst: Instances = x["instances"].to(device)
                # xyxy boxes -> xywh normalized (YOLO format)
                h, w = x["height"], x["width"]
                b = inst.gt_boxes.tensor.clone()
                b[:, 2:] = b[:, 2:] - b[:, :2]   # xyxy -> xywh
                b[:, 0] = b[:, 0] + b[:, 2]/2.0
                b[:, 1] = b[:, 1] + b[:, 3]/2.0
                b[:, [0,2]] /= w
                b[:, [1,3]] /= h
                cls = inst.gt_classes[:, None].float()
                yolo_targets = torch.cat([cls, b], dim=1)
                labels.append(yolo_targets)

            preds = self.backbone(images)
            loss_items = self.criterion(preds, labels)

            return {
                "loss_box": loss_items[0],
                "loss_obj": loss_items[1],
                "loss_cls": loss_items[2],
            }

        else:
            # Inference

            results = self.backbone.predict(images, conf=self.score_thresh, verbose=False, device=self.device)

            outputs = []
            for inp, r in zip(batched_inputs, results):
                inst = Instances((inp["height"], inp["width"]))
                boxes   = torch.tensor(r.boxes.xyxy.cpu().numpy())
                scores  = torch.tensor(r.boxes.conf.cpu().numpy())
                classes = torch.tensor(r.boxes.cls.cpu().numpy(), dtype=torch.int64)

                # top-K
                if len(scores) > self.max_det:
                    idx = torch.topk(scores, self.max_det).indices
                    boxes, scores, classes = boxes[idx], scores[idx], classes[idx]

                inst.pred_boxes   = Boxes(boxes)
                inst.scores       = scores
                inst.pred_classes = classes
                outputs.append({"instances": inst})
            return outputs
        
class YOLOTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=minmax_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=minmax_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        opt = super().build_optimizer(cfg, model)

        return model

    # @classmethod
    # def build_optimizer(cfg, model):
    #     return super().build_optimizer(cfg, model)