import torch
import torch.nn as nn

from detectron2.modeling import META_ARCH_REGISTRY, build_model
from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.evaluation import COCOEvaluator

# from ultralytics.nn.modules.block import C3k2, C3k, Bottleneck
# from ultralytics.nn.modules.block import SPPF, C2PSA, PSABlock
# from ultralytics.nn.modules.conv import Conv
# from ultralytics.nn.modules.head import Detect

from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss

from .utils import minmax_mapper

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

class Yolo_Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        import ultralytics
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")

        self.conv1 = yolo.model.model[0]
        self.conv2 = yolo.model.model[1]
        self.c3k2_1 = yolo.model.model[2]
        self.conv3 = yolo.model.model[3]
        self.c3k2_2 = yolo.model.model[4]
        self.conv4 = yolo.model.model[5]
        self.c3k2_3 = yolo.model.model[6]
        self.conv5 = yolo.model.model[7]
        self.c3k2_4 = yolo.model.model[8]

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3k2_1(x)

        x = self.conv3(x)
        x_8 = self.c3k2_2(x)
        
        x = self.conv4(x)
        x_16 = self.c3k2_3(x)
        
        x = self.conv5(x)
        x_32 = self.c3k2_4(x)

        return x_8, x_16, x_32


class Yolo_Neck(nn.Module):

    def __init__(self):
        super().__init__()

        import ultralytics
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")

        self.sppf = yolo.model.model[9]
        self.c2psa = yolo.model.model[10]

        self.us1 = yolo.model.model[11]
        self.concat_1 = yolo.model.model[12]
        self.c3k2_1 = yolo.model.model[13]
        self.us2 = yolo.model.model[14]
        self.concat_2 = yolo.model.model[15]
        self.c3k2_2 = yolo.model.model[16]

        self.conv1 = yolo.model.model[17]
        self.concat_3 = yolo.model.model[18]
        self.c3k2_3 = yolo.model.model[19]
        self.conv2 = yolo.model.model[20]
        self.concat_4 = yolo.model.model[21]
        self.c3k2_4 = yolo.model.model[22]

    def forward(self, x_8, x_16, x_32):

        x_32 = self.sppf(x_32)
        x_32 = self.c2psa(x_32)
        
        x = self.us1(x_32)
        x = self.concat_1([x, x_16])
        x_16 = self.c3k2_1(x)

        x = self.us2(x_16)
        x = self.concat_2([x, x_8])
        x_8 = self.c3k2_2(x)

        x = self.conv1(x_8)
        x = self.concat_3(x_8, x_16)
        x_16 = self.c3k2_3(x)

        x = self.conv2(x_16)
        x = self.concat_4(x, x_32)
        x_32 = self.c3k2_4(x)

        return x_8, x_16, x_32


@META_ARCH_REGISTRY.register()
class MyYolo(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Yolo_Backbone()
        self.neck = Yolo_Neck()

        self.det_head = self._get_det_head()

        self.criterion = v8DetectionLoss(self.det_head)

    def _get_det_head(self):
        import ultralytics
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")

        return yolo.model.model[23]

    def forward(self, x):
        x_8, x_16, x_32 = self.backbone(x)

        x_8, x_16, x_32 = self.neck(x_8, x_16, x_32)

        x = self.det_head([x_8, x_16, x_32])

        return x

        
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