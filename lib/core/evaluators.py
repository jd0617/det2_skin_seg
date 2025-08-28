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

            vis = Visualizer(
                img[:, :, ::-1], metadata=self.meta, scale=1.0, instance_mode=ColorMode.IMAGE
            )
            drawn = vis.draw_instance_predictions(inst).get_image()[:, :, ::-1]
            # name file by image_id if available
            stem = str(inp.get("image_id", os.path.basename(inp["file_name"])))
            cv2.imwrite(os.path.join(self.output_dir, f"{stem}.jpg"), drawn)
            self._count += 1

    def evaluate(self):
        # nothing to aggregate; just a side effect
        return {}

    # def vis_batch(self, inputs, outputs, save_path):

    #     # assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"

    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     fontScale = 1.0
    #     clr = (255, 0, 0)
    #     thickness = 2

    #     h = self.cfg.DATASET.IMG_SIZE[1]
    #     w = self.cfg.DATASET.IMG_SIZE[0]

    #     batch_size = len(outputs)

    #     x = w//4 + 50

    #     put_text = partial(cv2.putText, org=(x, 60), fontFace=font, fontScale=fontScale, color=clr, thickness=thickness)
    #     nnf_pad = partial(nnf.pad, pad=(5, 5, 5, 5), mode='constant', value=255)

    #     img_root = self.cfg.DATASET.IMG_DIR

    #     stacked_output = []

    #     for input, output in tqdm(zip(inputs, outputs)):
    #         img_path = input["file_name"]
    #         img_name = os.path.basename(img_path)

    #         ori_img = cv2.imread(img_path)
    #         ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    #         h, w, c = ori_img.shape

    #         # ori_img = img.transpsoe(2, 0, 1) # (c, h, w)
    #         # ori_img = T.ToTensor()(ori_img)

    #         gt = input["instances"].gt_masks.tensor  # (num_gt, H, W)
    #         pred = output["instances"].pred_masks

    #         gt = gt.cpu()
    #         pred = pred.cpu()

    #         if pred.shape[2] != img.shape[2]:
    #             pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    #         if gt.shape != img.shape:
    #             gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    #         gt = torch.clamp(gt*255., min=0., max=255.) # c, h, w
    #         if gt.size(1) < h:
    #             gt = nnf.interpolate(batch_gt, size=(h, w), mode='bilinear', align_corners=False)

    #         pred = torch.clamp(batch_pred*255., min=0., max=255.)
    #         if pred.size(1) < h:
    #             pred = nnf.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)

    #         ori_img = put_text(ori_img, img_name)
    #         ori_img = torch.from_numpy(ori_img).permute(2, 0, 1)
            
    #         ori_img = nnf_pad(ori_img)
    #         gt = nnf_pad(gt)
    #         pred = nnf_pad(pred)
    #         _, ph, pw = preds.shape

    #         gt = torch.cat(gt.chunk(15, dim=0), dim=1)
    #         gt = gt.repeat(3, 1, 1) # [3, h, w*b]

    #         pred = torch.cat(pred.chunk(15, dim=0), dim=1)
    #         pred = pred.repeat(3, 1, 1) # [3, h, w*b]
    #         pred = torch.cat([torch.zeros([3, ph, pw]), preds], 2)

    #         img_gt = torch.cat([ori_img, gt], 2)
    #         img_gt_pred = torch,cat([img_gt, pred], 1)

    #         stacked_output.append(img_gt_pred)

    #     stacked_output = torch.stack(stacked_output)

    #     grid = torchvision.utils.make_grid(stacked_output, nrow=1, padding=5, pad_value=255)
    #     grid = grid.byte().permute(1, 2, 0).cpu().numpy()
    #     grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    #     cv2.imwrite(save_path, grid)




