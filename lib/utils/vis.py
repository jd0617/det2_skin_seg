from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import torch
import cv2
import torchvision.transforms as T
import torch.nn.functional as nnf
import os

from pathlib import Path
from functools import partial
from torchvision.utils import make_grid

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color_gt = (255, 0, 0)
thickness = 2

def save_seg_results_gpu(batch_img, batch_gt, batch_pred, file_name, img_names=None, nrow=16, padding=5):
    batch_img = batch_img.detach()
    batch_gt = batch_gt.detach()
    batch_pred = batch_pred.detach()

    batch_img = (batch_img*255).byte()
    batch_gt = (batch_gt*255).byte()
    batch_pred = (batch_pred*255).byte()

    if img_names is not None:

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.0
        clr = (255, 0, 0)
        thickness = 2

        x = 10 #batch_img.size(3)//4 + 50
        y = 50

        put_text = partial(cv2.putText, org=(x, y), fontFace=font, fontScale=fontScale, color=clr, thickness=thickness)
        
        batch_img = batch_img.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        nbatch_img = np.zeros_like(batch_img)

        for idx, (img, imgn) in enumerate(zip(batch_img, img_names)):
            img = put_text(img, imgn)

            nbatch_img[idx, :, :, :] = img

        batch_img = torch.from_numpy(nbatch_img).to(batch_pred.device).permute(0, 3, 1, 2)

    if batch_gt.dim() == 3:
        batch_gt = batch_gt.unsqueeze(1)

    if batch_pred.dim() == 3:
        batch_pred = batch_pred.unsqueeze(1)

    pad = (5,5,0,0)

    # batch_img = nnf.pad(batch_img, (0, 0, ), mode="constant", value=255)

    nsize = (batch_img.size(2), batch_img.size(3))

    batch_gt = nnf.interpolate(batch_gt, size=nsize, mode="nearest")
    batch_pred = nnf.interpolate(batch_pred, size=nsize, mode="nearest")

    batch_gt = nnf.pad(batch_gt, pad, mode="constant", value=255)
    batch_pred = nnf.pad(batch_pred, pad, mode="constant", value=255)

    batch_gt = batch_gt.repeat(1, 3, 1, 1)
    batch_pred = batch_pred.repeat(1, 3, 1, 1)

    stacked_batch = torch.cat([batch_img, batch_pred, batch_gt], dim=3)

    grid = make_grid(stacked_batch, nrow, padding, False, pad_value=255.)
    grid = grid.byte().permute(1, 2, 0).cpu().numpy()
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    cv2.imwrite(file_name, grid)



def save_segmentation_results(batch_img, batch_gt, batch_pred, file_name, img_names, nrow=3, padding=2): #, norm=True
    
    batch_img = batch_img.detach().cpu().numpy()

    if batch_pred.ndim > 3:
        batch_pred = batch_pred.squeeze()

    pad = partial(cv2.copyMakeBorder, top=3, bottom=3, left=10, right=10, borderType=cv2.BORDER_CONSTANT)

    pad_img = partial(cv2.copyMakeBorder, top=3, bottom=3, left=2, right=2, borderType=cv2.BORDER_CONSTANT)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    clr = (255, 0, 0)
    thickness = 2

    b, c, h, w = batch_img.shape

    x = w//4 + 50

    put_text = partial(cv2.putText, org=(x, 60), fontFace=font, fontScale=fontScale, color=clr, thickness=thickness)

    stacked_igp = []


    if pred.shape[2] != img.shape[2]:
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if gt.shape != img.shape:
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    batch_gt = np.clip(batch_gt*255., 0., 255.)
    batch_gt = np.repeat(batch_gt, 3, axis=-1)
    batch_gt = np.transpose(batch_gt, (1, 2, 0))
    if len(batch_gt.shape) > 3:
        batch_gt = batch_gt.squeeze()

    batch_pred = np.clip(batch_pred*255., 0., 255.)
    batch_pred = np.repeat(batch_pred, 3, axis=-1)
    batch_pred = np.transpose(batch_pred, (1, 2, 0))
    if len(batch_pred.shape) > 3:
        batch_pred = batch_pred.squeeze()


    for (img, gt, pred, imgn) in zip(batch_img, batch_gt, batch_pred, img_names):
        
        # if pred.shape != img.shape:
        #     pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # if gt.shape != img.shape:
        #     gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))

        img = put_text(img, imgn)
        
        gt = pad(gt, value=[0, 255, 0])
        img = pad_img(img, value=[0, 0, 0])
        pred = pad(pred, value=[0, 0, 255])

        igp = np.hstack([img, gt, pred])

        stacked_igp.append(igp)

    stacked_igp = torch.from_numpy(np.array(stacked_igp)).permute(0, 3, 1, 2)
    # stacked_igp = 

    grid = make_grid(stacked_igp, nrow, padding, False, pad_value=255.)
    grid = grid.byte().permute(1, 2, 0).cpu().numpy()
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    cv2.imwrite(file_name, grid)


def save_batch_img_with_mask(batch_img:torch.Tensor, batch_gt:torch.Tensor, batch_pred:torch.Tensor, file_name, imgn_list, nrow=1, padding=2, normalize=False):
    # if normalize:
    #     batch_img = batch_img.clone()
    #     min = float(batch_img.min())
    #     max = float(batch_img.max())

        # batch_img.add_(-min).div_(max - min + 1e-5)

    # transform = T.Resize((batch_pred.shape[2], batch_pred.shape[3]))
    # batch_img = cv2.resize(batch_img, (batch_pred.shape[3], batch_pred.shape[2]))


    """
    pytorch version
    required shape for batch_img: [b, c, h, w]
    required shape for batch_gt: [b, c, h, w]
    required shape for batch_pred: [b, c, h, w]
    """
    batch_img = batch_img.detach().cpu()
    batch_gt = batch_gt.detach().cpu()
    batch_pred = batch_pred.detach().cpu()

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    clr = (255, 0, 0)
    thickness = 2

    b, h, w, c = batch_img.shape 

    x = w//4 + 50

    put_text = partial(cv2.putText, org=(x, 60), fontFace=font, fontScale=fontScale, color=clr, thickness=thickness)

    batch_gt = torch.clamp(batch_gt*255., min=0., max=255.) # b, c, h, w
    if batch_gt.size(2) < h:
        batch_gt = nnf.interpolate(batch_gt, size=(h, w), mode='bilinear', align_corners=False)
    
    batch_pred = torch.clamp(batch_pred*255., min=0., max=255.)
    if batch_pred.size(2) < h:
        batch_pred = nnf.interpolate(batch_pred, size=(h, w), mode='bilinear', align_corners=False)

    imgs = []

    for (img, imgn) in zip(batch_img, imgn_list):
        img = img.squeeze().numpy()
        img = put_text(img, imgn)

        imgs.append(img)

    batch_img = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)


    stacked_igp = []

    nnf_pad = partial(nnf.pad, pad=(5, 5, 5, 5), mode='constant', value=255)

    for (img, gts, preds, imgn) in zip(batch_img, batch_gt, batch_pred, imgn_list):

        img = nnf_pad(img)

        gts = nnf_pad(gts)
        gts = torch.cat(gts.chunk(15, dim=0), dim=2) # [1, h, w*c]
        gts = gts.repeat(3, 1, 1) # [3, h, w*b]

        preds = nnf_pad(preds)
        _, ph, pw = preds.shape
        preds = torch.cat(preds.chunk(15, dim=0), dim=2) # [1, h, w*c]
        preds = preds.repeat(3, 1, 1) # [3, h, w*b]

        preds = torch.cat([torch.zeros([3, ph, pw]), preds], 2)

        img_gts = torch.cat([img, gts], 2)
        img_gts_preds = torch.cat([img_gts, preds], 1)

        stacked_igp.append(img_gts_preds) #.unsqueeze(0)

    stacked_igp = torch.stack(stacked_igp)
    #     # stacked_igp = stacked_igp.permute(0, 3, 1, 2)
    #
    grid = torchvision.utils.make_grid(stacked_igp, nrow, padding, False, pad_value=255.)
    ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
    # ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    cv2.imwrite(file_name, ndarr)




# def save_batch_img_with_mask(batch_img:torch.Tensor, batch_gt:torch.Tensor, batch_pred:torch.Tensor, file_name, imgn_list, nrow=1, padding=2, normalize=False):
#     # if normalize:
#     #     batch_img = batch_img.clone()
#     #     min = float(batch_img.min())
#     #     max = float(batch_img.max())

#         # batch_img.add_(-min).div_(max - min + 1e-5)

#     # transform = T.Resize((batch_pred.shape[2], batch_pred.shape[3]))
#     # batch_img = cv2.resize(batch_img, (batch_pred.shape[3], batch_pred.shape[2]))


#     """
#     pytorch version
#     required shape for batch_img: [b, c, h, w]
#     required shape for batch_gt: [b, c, h, w]
#     required shape for batch_pred: [b, c, h, w]
#     """
#     batch_img = batch_img.detach().cpu()
#     batch_gt = batch_gt.detach().cpu()
#     batch_pred = batch_pred.detach().cpu()

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1.0
#     clr = (255, 0, 0)
#     thickness = 2

#     b, c, h, w = batch_img.shape 

#     x = w//4 + 50

#     put_text = partial(cv2.putText, org=(x, 60), fontFace=font, fontScale=fontScale, color=clr, thickness=thickness)

#     batch_gt = np.clip(batch_gt*255., 0., 255.) # b, c, h, w
#     if batch_gt.size(2) < h:
#         batch_gt = cv2.resize(batch_gt, (w,h))

#     batch_pred = np.clip(batch_pred*255., 0., 255.)
#     batch_pred = cv2.resize(batch_pred, (w,h))

#     stacked_igp = []

#     for (img, gts, preds, imgn) in zip(batch_img, batch_gt, batch_pred, imgn_list):
        
#         img = put_text(img, imgn).squeeze()

#         gts = np.repeat(gts, 3, axis=1)
#         preds = np.repeat(preds, 3, axis=1)

#         img = np.expand_dims(img, 0) # [1, c, h, w]
#         img_gts = np.concatenate((img, gts), axis=0)
#         img_preds = np.concatenate((np.zeros_like(img), preds), axis=0)
#         # two_col = np.transpose(np.concatenate((img_gts, img_preds), axis=0), (0, 3, 1, 2))
#         two_col = torch.from_numpy(np.concatenate((img_gts, img_preds), axis=0))

#         stacked_igp.append(
#             torchvision.utils.make_grid(two_col,
#                                         two_col.shape[0]//2,
#                                         padding,
#                                         False, pad_value=255.).numpy())

#     stacked_igp = torch.from_numpy(np.array(stacked_igp))
#     #     # stacked_igp = stacked_igp.permute(0, 3, 1, 2)
#     #
#     grid = torchvision.utils.make_grid(stacked_igp, nrow, padding, False, pad_value=255.)
#     ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
#     # ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
#     ndarr = ndarr.copy()
#     ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

#     cv2.imwrite(file_name, ndarr)

