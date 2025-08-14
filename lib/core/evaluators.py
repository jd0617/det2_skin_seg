from detectron2.evaluation import DatasetEvaluator
from torchmetrics.functional import dice

import logging
import torch

from .metrics import dice_score_np, dice_score_calc, get_confusion_metrics
from .vis import save_batch_img_with_mask

logger = logging.getLogger(__name__)

class MyEvaluator(DatasetEvaluator):
    def __init__(self, from_logits=True, threshold=0.5):
        self.results = []
        self.from_logits = from_logits
        self.threshold = threshold

    def reset(self):
        self.results = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt_masks = input["instances"].gt_masks.tensor  # (num_gt, H, W)
            pred_masks = output["instances"].pred_masks

            assert gt_mask.s

            if self.from_logits:
                pred_masks = torch.sigmoid(pred_masks)
            
            pred_masks = (pred_masks > self.threshold).float()

            dice_score = dice(pred_masks, gt_masks.int())

            self.results.append(dice_score)

    def evaluate(self):
        if len(self.results) == 0:
            return {"Dice": float("nan")}
        return {"Dice": sum(self.results) / len(self.results)}
    








        



