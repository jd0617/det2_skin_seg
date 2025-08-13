import numpy as np

def dice_score_np(pred, target, eps=1e-6):
    """
    pred, target: numpy arrays of shape [B, H, W] or [B, 1, H, W]
    Assumes binary masks (0 or 1)
    """
    if pred.ndim == 4:
        pred = pred[:, 0]  # remove channel dim if present
        target = target[:, 0]

    B = pred.shape[0]
    dice_scores = []

    for i in range(B):
        p = pred[i].astype(np.bool_)
        t = target[i].astype(np.bool_)
        intersection = np.logical_and(p, t).sum()
        union = p.sum() + t.sum()
        dice = (2. * intersection + eps) / (union + eps)
        dice_scores.append(dice)

    return np.mean(dice_scores)

def dice_score_calc(pred, target, eps=1e-6):

    pred = pred.detach()
    target = target.detach()

    pred = pred.view(pred.size(0), -1)  # (B, H*W)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice_score = (2 * intersection + eps) / (union / eps)

    return dice_score.mean()


def get_confusion_metrics(pred, target):

    true_pred = target == pred
    false_pred = target != pred
    pos_pred = pred == 1
    neg_pred = pred == 0

    tp = np.sum(true_pred * pos_pred)
    fp = np.sum(false_pred * pos_pred)
    
    tn = np.sum(true_pred * neg_pred)
    fn = np.sum(false_pred * neg_pred)

    return tp, fp, tn, fn
