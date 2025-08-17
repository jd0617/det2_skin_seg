import os

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone, BACKBONE_REGISTRY, Backbone

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from lib.core.loss import get_loss_fn
from ..models.hrnet import HighResolutionNet

BN_MOMENTUM = 0.1

@META_ARCH_REGISTRY.register()
class HRNet_SEG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_masks = cfg.MODEL.NUM_MASKS
        self.align_corners = cfg.MODEL.ALIGN_CORNERS

        self.backbone = build_hrnet_backbone(cfg)

        fm_c = sum(cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS)

        self.seg_head = nn.Sequential(
            nn.Conv2d(fm_c, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_masks, 1, stride=1, padding=0),
        )

        self.criteion = self._make_criterion(cfg)

        self.init_weights(cfg.MODEL.PRETRAINED)

    def _make_criterion(self, cfg):
        return get_loss_fn(cfg)

    
    def forward(self, x):

        x = torch.stack([xe['image'] for xe in x])
        x = self.backbone(x)

        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = nnf.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x2 = nnf.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x3 = nnf.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.seg_head(x)

        if self.training:
            gt = torch.stack([xe['instance'].gt_masks.tensor for xe in x])
            loss = self.criterion(x, gt)
            output = {'loss': loss}
        else:
            output = {'mask_logits': x}

        return output

    def init_weights(self, pretrained=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            print('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

@BACKBONE_REGISTRY.register()
def build_hrnet_backbone(cfg, input_c=3):
    backbone = HighResolutionNet(cfg, input_c)

    return backbone