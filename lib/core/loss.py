import torch
import torch.nn as nn
import torch.nn.functional as nnf
from scipy.ndimage import distance_transform_edt
import numpy as np

import logging

logger = logging.getLogger(__name__)

class DiceBoundaryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, eps=1e-6, **kwargs):
        super().__init__()

        # self.alpha = kwargs.get("alpha", 1.0)
        # self.beta = kwargs.get("beta", 1.0)
        # self.eps = kwargs.get("smooth", 1e-6)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        logger.info("DiceBoundaryLoss")
        logger.info(f"===> alpha: {self.alpha}")
        logger.info(f"===> beta: {self.beta}")
        logger.info(f"===> eps: {self.eps}")

    def dice_loss(self, pred, target):
        intersection = 2 * (pred*target).sum(dim=(1,2,3))
        union = (pred**2).sum(dim=(1,2,3)) + (target**2).sum(dim=(1,2,3))
        loss = 1-(intersection+self.eps) / (union+self.eps)
        return loss.mean()
    
    def compute_boundary_map(self, mask):
        mask = mask.detach().cpu().numpy().astype(np.uint8)
        dist_maps = []
        for m in mask:
            m = m[0]
            posmask = m.astype(np.bool_)
            negmask = ~posmask
            dist_out = distance_transform_edt(negmask)
            dist_in = distance_transform_edt(posmask)
            dist_map = dist_out + dist_in
            dist_maps.append(dist_map)
        dist_maps = np.stack(dist_maps)  # (B, H, W)
        return torch.tensor(dist_maps).unsqueeze(1).float()  # (B, 1, H, W)
    
    def boundary_loss(self, pred, target):
        dist_map = self.compute_boundary_map(target).to(pred.device)
        return (pred * dist_map).mean()

    def forward(self, pred, target, from_logits=False):
        """
        pred: raw logits, shape (B, 1, H, W)
        target: binary mask, shape (B, 1, H, W)
        """
        if target.dim() < 4:
            target = target.unsqueeze(1)
        
        if from_logits:
            pred = nnf.sigmoid(pred)
            
        d_loss = self.dice_loss(pred, target)
        b_loss = self.boundary_loss(pred, target)
        return self.alpha * d_loss + self.beta * b_loss
    

class MultiLoss(nn.Module):
    def __init__(self, cls_loss="regression", alpha=1.0, beta=1.0, eps=1e-6):
        self.cls_loss = cls_loss
        self.dice_boundary_loss = DiceBoundaryLoss(
            alpha=alpha, beta=beta, eps=eps
        )
        if cls_loss == "regression":
            self.cell_loss = nn.MSELoss()
        else:
            self.cell_loss = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_loss = self.dice_boundary_loss(seg_pred, seg_target)
        cls_loss = self.cell_loss(cls_pred, cls_target)

        loss = seg_loss + cls_loss

        return loss
    

# class FocalDiceLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", smooth=1e-5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.smooth = smooth

#     def dice_loss(self, pred, target):
#         b, c, h, w = pred.shape
#         pred = pred.view(b, c, -1)
#         target = target.view(b, c, -1)

#         intersection = torch.sum(pred * target, -1)
#         union = torch.sum(pred, -1) + torch.sum(target, -1) + self.smooth

#         return (2.0 * intersection + self.smooth) / union

#     def forward(self, pred, target, from_logits=False):

#         if from_logits:
#             pred = nnf.sigmoid(pred)

#         dice_loss = self.dice_loss(pred, target)
#         # shape: [b, c]

#         p_t = (target * pred) + ((1 - target) * (1 - pred))
#         alpha_factor = 1.0
#         modulating_factor = 1.0

#         if self.alpha:
#             alpha = torch.tensor(self.alpha, dtype=target.dtype)
#             alpha_factor = target * alpha + (1 - target) * (1 - alpha)
#             alpha_factor = alpha_factor.mean(-1).mean(-1)

#         if self.gamma:
#             gamma = torch.tensor(self.gamma, dtype=target.dtype)
#             modulating_factor = torch.pow((1.0 - p_t), gamma)
#             modulating_factor = modulating_factor.mean(-1).mean(-1)

#         output = alpha_factor * modulating_factor * dice_loss

#         if self.reduction == "sum":
#             loss = 1 - output.sum(1).sum()
#         else:
#             loss = 1 - output.mean(1).mean()

#         return loss
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth #kwargs.get("smooth", 1e-6)
        
        logger.info("DiceLoss")
        logger.info(f"===> smooth: {self.smooth}")

    def forward(self, preds, targets, from_logits=False):
        # If preds are logits, apply sigmoid
        if from_logits:
            preds = nnf.sigmoid(preds)
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Compute Dice score
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-5, **kwargs):
        super(FocalDiceLoss, self).__init__()

        self.alpha = alpha #kwargs.get("alpha", 0.25)
        self.gamma = gamma #kwargs.get("gamma", 2.0)
        self.reduction = reduction #kwargs.get("reduction", 'mean')
        self.eps = eps #kwargs.get("smooth", 1e-5)

        logger.info("FocalDiceLoss")
        logger.info(f"===> alpha: {self.alpha}")
        logger.info(f"===> gamma: {self.gamma}")
        logger.info(f"===> beta: {self.beta}")
        logger.info(f"===> eps: {self.eps}")

    def dice_loss(self, preds, targets):
        preds = preds.view(preds.size(0), -1)  # (B, H*W)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.eps) / (union / self.eps)

        # dice = 1 - dice

        return dice

    def forward(self, preds, targets, from_logits=False):
        if from_logits:
            preds = nnf.Sigmoid(preds)

        dice_loss = self.dice_loss(preds, targets)

        p_t = (targets * preds) + ((1 - targets) * (1 - preds))
        alpha_factor = 0
        modulating_factor = 0

        if self.alpha:
            alpha = torch.tensor(self.alpha, dtype=targets.dtype)
            alpha_factor = targets * alpha + (1-targets) - (1-alpha)
            alpha_factor = alpha_factor.mean(-1).mean(-1)

        if self.gamma:
            gamma = torch.tensor(self.gamma, dtype=targets.dtype)
            modulating_factor = torch.pow((1.0-p_t), gamma)
            modulating_factor = modulating_factor.mean(-1).mean(-1)

        loss = alpha_factor * modulating_factor * dice_loss

        loss = loss.sum() if self.reduction == "sum" else loss.mean()

        return 1 - loss

        
class DiceLossWithBCE(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, **kwargs):
        super().__init__()
        self.alpha = alpha #kwargs.get("alpha", 0.5)
        self.beta = beta #kwargs.get("beta", 0.5)
        self.smooth = smooth #kwargs.get("smooth", 1e-6)

        self.bce_fn = nn.BCEWithLogitsLoss()

        logger.info("DiceLossWithBCE")
        logger.info(f"===> alpha: {self.alpha}")
        logger.info(f"===> beta: {self.beta}")
        logger.info(f"===> eps: {self.smooth}")

    def dice_loss_fn(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute Dice score
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def forward(self, pred, target, from_logits=True):

        if target.dim() < pred.dim():
            target = target.unsqueeze(1)

        if from_logits:
            dice_loss = self.dice_loss_fn(nnf.sigmoid(pred), target)
        else:
            dice_loss = self.dice_loss_fn(pred, target)

        # with torch.autocast(device_type="cuda", enabled=False):

        bce_loss = self.bce_fn(pred, target)

        loss = self.alpha*dice_loss + self.beta*bce_loss

        return loss    


class DiceFocalBCE(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.5, smooth=1e-6, reduction='mean', dice_rat=1.0): #**kwargs): #
        super().__init__()
        self.alpha = alpha #kwargs.get("alpha", 0.25)
        self.gamma = gamma #kwargs.get("gamma", 2.0)
        self.smooth = smooth #kwargs.get("smooth", 1e-6)
        self.reduction = reduction #kwargs.get("reduction", 'mean')
        # dice_rat = kwargs.get("dice_rat", 1.0)
        self.dice_rat = dice_rat
        self.fbce_rat = 1.0 if dice_rat==1.0 else abs(1.0-dice_rat)

        logger.info("DiceFocalBCE")
        logger.info(f"===> alpha: {self.alpha}")
        logger.info(f"===> gamma: {self.gamma}")
        logger.info(f"===> smooth: {self.smooth}")
        logger.info(f"===> reduction: {self.reduction}")
        logger.info(f"===> dice_rat: {self.dice_rat}")
        logger.info(f"===> fbce_rat: {self.fbce_rat}")

    def dice_loss_fn(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute Dice score
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
    def focal_bce_fn(self, pred, target):
        bce_loss = nnf.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt = torch.where(target == 1, pred, 1-pred)
        focal_weight = (1-pt) ** self.gamma
        fbce = self.alpha * focal_weight * bce_loss

        if self.reduction == "sum":
            fbce = fbce.sum()
        else:
            fbce = fbce.mean()

        return fbce

    def forward(self, pred, target, from_logits=True):

        if target.dim() < pred.dim():
            target = target.unsqueeze(1)

        if from_logits:
            dice_loss = self.dice_loss_fn(nnf.sigmoid(pred), target)
        else:
            dice_loss = self.dice_loss_fn(pred, target)

        fbce_loss = self.focal_bce_fn(pred, target)

        loss = self.dice_rat*dice_loss + self.fbce_rat*fbce_loss

        return loss


def get_loss_fn(cfg):
    loss_cfg = cfg.LOSS
    loss_name = cfg.LOSS.LOSS

    args = {}

    for k in loss_cfg:
        v = loss_cfg[k]

        args[k.lower()] = v

    criterion = eval(loss_name)() #**args

    logger.info("=> criterion used: {}".format(criterion._get_name()))

    return criterion
