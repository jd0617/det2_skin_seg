# Project File Structure
## Dataset
Diagram below shows only neccesary folders.

```
${seg_}
 `-- preprocessed
     |-- anno
     |   `-- coco_format
     |       |-- all_complete_merge.json
     |       |-- all_complete_train.json # not use while doing k-fold
     |       `-- all_complete_test.json # not use while doing k-fold
     `-- pos_cropped_patch_all_r1_r2
         |-- data_image_0.jpg
         |-- data_image_1.jpg
         |-- data_image_2.jpg
         |-- data_image_3.jpg
```

# Train

python tools/train_nested_kfold.py --cfg /workspace/project/configs/hrnet/w32_ori.yaml

