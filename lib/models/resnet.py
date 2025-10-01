from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.resnet import ResNet, BasicBlock, ResNetBlockBase, make_stage
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

@BACKBONE_REGISTRY.register()
def build_resnet18_backbone(cfg, input_shape):
    # ResNet-18 structure: [2, 2, 2, 2]
    return ResNet(
        stem=ResNetBlockBase(cfg.MODEL.RESNETS.STEM_OUT_CHANNELS, norm=cfg.MODEL.RESNETS.NORM),
        stages=[
            make_stage(BasicBlock, 64, 64, 2, stride=1),
            make_stage(BasicBlock, 64, 128, 2, stride=2),
            make_stage(BasicBlock, 128, 256, 2, stride=2),
            make_stage(BasicBlock, 256, 512, 2, stride=2),
        ],
        num_classes=None,  # we only want features, not classification
        out_features=cfg.MODEL.RESNETS.OUT_FEATURES,
    )
