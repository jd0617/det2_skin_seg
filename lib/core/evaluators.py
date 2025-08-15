from detectron2.evaluation import DatasetEvaluator

import logging
import torch
import torchvision.transforms as T

from tqdm import tqdm

from .metrics import dice_score_np, dice_score_calc, get_confusion_metrics
from .vis import save_batch_img_with_mask

try:
    from torchmetrics.functional import dice
except ImportError:
    frp,m 


logger = logging.getLogger(__name__)

class MyEvaluator(DatasetEvaluator):
    def __init__(self, cfg, from_logits=True, threshold=0.5, vis_dir=None):
        self.results = []

        self.cfg = cfg
        self.from_logits = from_logits
        self.threshold = threshold
        self.vis_dir = vis_dir

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

    def evaluate(self):
        if len(self.results) == 0:
            return {"Dice": float("nan")}
        return {"Dice": sum(self.results) / len(self.results)}

    def vis_batch(self, inputs, outputs):

        assert len(inputs) == len(outputs), "Inputs and outputs must have the same length"

        pad = partial(cv2.copyMakeBorder, top=3, bottom=3, left=10, right=10, borderType=cv2.BORDER_CONSTANT)

        pad_img = partial(cv2.copyMakeBorder, top=3, bottom=3, left=2, right=2, borderType=cv2.BORDER_CONSTANT)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.0
        clr = (255, 0, 0)
        thickness = 2

        h = self.cfg.DATASET.IMG_SIZE[1]
        w = self.cfg.DATASET.IMG_SIZE[0]

        batch_size = len(outputs)

        x = w//4 + 50

        put_text = partial(cv2.putText, org=(x, 60), fontFace=font, fontScale=fontScale, color=clr, thickness=thickness)

        img_root = self.cfg.DATASET.IMG_DIR

        batch_output = []

        for input, output in tqdm(zip(inputs, outputs)):
            img_path = inpue["file_name"]

            ori_img = cv2.imread(img_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            ori_img = img.transpsoe(2, 0, 1) # (c, h, w)
            # ori_img = T.ToTensor()(ori_img)

            gt = input["instances"].gt_masks.tensor  # (num_gt, H, W)
            pred = output["instances"].pred_masks

            gt = gt.cpu().numpy()
            pred = pred.cpu().numpy()

            if pred.shape[2] != img.shape[2]:
                pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            if gt.shape != img.shape:
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            gt = np.clip(batch_gt*255., 0., 255.)
            gt = np.repeat(gt, 3, axis=0)









