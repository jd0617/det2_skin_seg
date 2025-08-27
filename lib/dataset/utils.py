import os
import logging

import numpy as np

from pathlib import Path
from functools import partial

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from pycocotools.coco import COCO
from sklearn.model_selection import GroupKFold

from copy import deepcopy

logger = logging.getLogger(__name__)


def load_coco(json_file, img_root):
    ds_dicts = load_coco_json(json_file, img_root) # detectron2 style

    coco = COCO(json_file) # pycocotools style

    ids = {img_id: coco.imgs[img_id].get("patient_id") for img_id in coco.imgs}

    out = []

    # combine "patient_id" from coco to ds_dicts
    for d in ds_dicts:
        d = deepcopy(d)
        img_id = d["image_id"]
        d["patient_id"] = ids[img_id]
        out.append(d)

    return out


def get_patient_id(anno):
    return str(anno["patient_id"])

def register_dataset(name: str, json_file: str, img_root: str, to_bin=False): #, coco_style:bool=False
    if name in DatasetCatalog.list():
        logger.info(f"Found existed {name}, removing existed {name}...")
        DatasetCatalog.remove(name)

    # if coco_style:
    #     logger.info(f"Registering {name} in COCO style...")
    #     register_cooc_instances(name, {}, json_file, img_root)
    # else:
    logger.info(f"Registering {name}...")
    DatasetCatalog.register(name, lambda: load_coco(json_file, img_root)) # , attach_pid_if_missing=True
    
    logger.info("Done!")

def to_bin(ds):
    out = []
    for d in ds:
        d = d.copy()
        anns = []
        for a in d.get('annotations', []):
            old = int(a["category_id"])
            if old > 1:
                a = a.copy()
                a["category_id"] = 1
            anns.append(a)
        d["annotations"] = anns
        out.append(d)
    return out


def register_patch_bin_dataset(name: str, json_file: str, img_root: str, extra_key=["patient_id"]):
    if name in DatasetCatalog.list():
        logger.info(f"Found existed {name}, removing existed {name}...")
        DatasetCatalog.remove(name)

    if len(extra_key) > 0:
        load_coco_json = partial(extra_annotation_keys=extra_key)

    def _loader():
        ds = load_coco_json(json_file, img_root, name, extra_annotation_keys=extra_key)
        out = to_bin(ds)
    
    logger.info(f"Registering {name}...")
    DatasetCatalog.register(name, _loader)


### K-Fold related functions

def get_records(name: str):
    return DatasetCatalog.get(name)

def get_groups_from_records(records):
    return np.array([str(r.get("patient_id")) for r in records])

def get_patient_ids(records) -> dict:
    ptn_ids = {}
    tmp = []

    idx = 0

    for i in records:
        ptn_id = i.get("patient_id")
        if ptn_id not in ptn_ids:
            ptn_ids[idx] = ptn_id
            idx += 1
            tmp.append(ptn_id)
        
    return ptn_ids

def group_kfold_indices(groups, n_splits=5, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    
    # map each sample to the shuffled order
    order = {g:i for i,g in enumerate(uniq)}
    order_idx = np.array([order[g] for g in groups])

    gkf = GroupKFold(n_splits=n_splits)
    # NOTE: gkf ignores y, uses groups only
    for tr, te in gkf.split(np.zeros_like(order_idx), groups=groups):
        yield tr, te

def register_split(name, base_records, indices):
    subset = [base_records[i] for i in indices]
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)

    DatasetCatalog.register(name, lambda s=subset: deepcopy(s))


