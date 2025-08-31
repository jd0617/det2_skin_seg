from detectron2.evaluation import DatasetEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

import logging
import torch
import torch.nn.functional as nnf
import torchvision.transforms as T

import cv2
import os
import numpy as np
from functools import partial

from tqdm import tqdm

from .metrics import dice_score_np, dice_score_calc, get_confusion_metrics
from .vis import save_batch_img_with_mask

try:
    from torchmetrics.functional import dice
except ImportError:
    from torchmetrics.functional.segmentation import dice_score
    dice = partial(dice_score, num_classes=2)


logger = logging.getLogger(__name__)



class DiceScoreEvaluator(DatasetEvaluator):
    def __init__(self, cfg, from_logits=True, threshold=0.5, mode:str='train'): # , vis=True, filename='vis.png'
        self.results = []

        self.cfg = cfg
        self.from_logits = from_logits
        self.threshold = threshold
        self.mode = mode.lower()
        self.vis_root = cfg.OUTPUT_DIR + "/vis_train" if mode == 'train' else cfg.OUTPUT_DIR + "/vis_test"

        # self.vis = vis
        # if vis:
        #     Path(self.vis_root).mkdir(parents=True, exist_ok=True)
        # self.filename = filename

    def reset(self):
        self.results = []

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            gt_masks = input["instances"].gt_masks.tensor  # (num_gt, H, W)
            pred_masks = output["instances"].pred_masks

            num_gt, num_pred = gt_masks.shape[0], pred_masks.shape[0]

            assert num_gt == num_pred, f"Mismatch: {num_gt} GT masks vs {num_pred} predicted masks"

            if self.from_logits:
                pred_masks = torch.sigmoid(pred_masks)
            
            pred_masks = (pred_masks > self.threshold).float()

            dice_score = dice(pred_masks, gt_masks.int())

            self.results.append(dice_score)

        # if self.vis:
        #     self.vis_batch(inputs, outputs, self.vis_root + "/vis_batch.png")


    def evaluate(self):
        if len(self.results) == 0:
            return {"dice_score": float("nan")}
        return {"dice_score": sum(self.results) / len(self.results)}
    

class VisualizeEval(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, max_images=50, score_thresh=None, topk=200):
        self.dataset_name = dataset_name
        self.output_dir   = os.path.join(output_dir, f"val_vis_{dataset_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_images   = max_images
        self.score_thresh = score_thresh
        self.topk         = topk
        self.meta         = MetadataCatalog.get(dataset_name)
        self._count       = 0

    def reset(self):
        self._count = 0

    def process(self, inputs, outputs):
        # inputs: list of dicts; outputs: list of dicts with "instances"
        for inp, out in zip(inputs, outputs):
            if self._count >= self.max_images:
                continue
            inst = out.get("instances", None)
            if inst is None:
                continue
            inst = inst.to("cpu")

            # optional filtering
            if self.score_thresh is not None and hasattr(inst, "scores"):
                keep = inst.scores >= float(self.score_thresh)
                inst = inst[keep]
            if self.topk and len(inst) > self.topk:
                inst = inst[: self.topk]

            # get the (transformed) image without re-reading from disk
            if "image" in inp:
                img = inp["image"].permute(1, 2, 0).numpy()  # HWC, BGR, float32 0..255
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = cv2.imread(inp["file_name"])  # fallback

            # Draw GT
            vis = Visualizer(
                img[:, :, ::-1], metadata=self.meta, scale=1.0, instance_mode=ColorMode.IMAGE
            )
            if "annotations" in inp:
                annos = []
                for a in inp["annotations"]:
                    b = {k: v for k, v in a.items() if k != "segmentation"}  # drop mask
                    annos.append(b)
                vis_img = vis.draw_dataset_dict({**inp, "annotations":annos})

            vis_img = vis.draw_instance_predictions(inst)
            drawn = vis_img.get_image()[:, :, ::-1]
            # name file by image_id if available
            stem = str(inp.get("image_id", os.path.basename(inp["file_name"])))

            drawn = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.output_dir, f"{stem}.jpg"), drawn)
            self._count += 1

    def evaluate(self):
        # nothing to aggregate; just a side effect
        return {}
