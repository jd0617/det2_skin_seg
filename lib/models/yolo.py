import torch, torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY, build_model
from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.evaluation import COCOEvaluator

from ultralytics.nn.modules.block import C3k2, C3k, Bottleneck
from ultralytics.nn.modules.block import SPPF, C2PSA, PSABlock
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

from ultralytics.utils.loss import v8DetectionLoss

class YOLO11n(nn.Module):
    super().__init__()

    def __init__(self, cfg, verbose=False):
                self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = {}  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        self.model_name = None  # model name

from .utils import minmax_mapper

        
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