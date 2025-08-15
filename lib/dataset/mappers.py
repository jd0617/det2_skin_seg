import cv2
import os
import logging

import torch
import detectron2.data.detection_utils as utils
from detefctron2.data import DatasetMapper
from detectron2.structures import Instaces

from pathlib import Path

logger = logging.getLogger(__name__)

class MyMapper:
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train

        self.img_root = cfg.DATASET.IMG_DIR
