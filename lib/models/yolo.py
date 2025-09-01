import torch, torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY, build_model
from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.evaluation import COCOEvaluator

from ultralytics import YOLO


from .utils import minmax_mapper

@META_ARCH_REGISTRY.register()
class UltralyticsYOLO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # load YOLOv8 model from Ultralytics
        # e.g. cfg.MODEL.YOLO.MODEL_NAME = "yolov8n.pt"
        self.model = YOLO(cfg.MODEL.EXTRA.MODEL_NAME)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        self.score_thresh = cfg.MODEL.EXTRA.SCORE_THRESH_TEST
        self.max_det      = cfg.TEST.DETECTIONS_PER_IMAGE

    def forward(self, batched_inputs):
        images = [x["image"].permute(1,2,0).cpu().numpy() for x in batched_inputs]

        if self.training:
            # convert Detectron2 Instances -> YOLO labels
            labels = []
            for x in batched_inputs:
                inst: Instances = x["instances"].to("cpu")
                # xyxy boxes -> xywh normalized (YOLO format)
                h, w = x["height"], x["width"]
                b = inst.gt_boxes.tensor.numpy()
                b[:, 2:] = b[:, 2:] - b[:, :2]   # xyxy -> xywh
                b[:, 0] = b[:, 0] + b[:, 2]/2.0
                b[:, 1] = b[:, 1] + b[:, 3]/2.0
                b[:, [0,2]] /= w
                b[:, [1,3]] /= h
                cls = inst.gt_classes.numpy()[:, None]
                yolo_targets = torch.tensor(np.hstack([cls, b]), dtype=torch.float32)
                labels.append(yolo_targets)

            # YOLOv8 training uses its own loop (not integrated here)
            raise NotImplementedError("Direct training inside Detectron2 not trivial — use YOLO’s trainer.")
        else:
            # Inference
            results = self.model.predict(images, conf=self.score_thresh, verbose=False, device=self.device)

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

        return model