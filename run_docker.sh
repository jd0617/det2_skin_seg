# docker run --gpus all -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvcr.io/nvidia/pytorch:24.12-py3 

DATA_PATH="/home/jd0617/Datasets_Models/Datasets/"
OUTPUT_DIR="/home/jd0617/Saved/"
MODEL_DIR="/home/jd0617/Datasets_Models/Models"

PROJ_DIR="./"

docker run -d --gpus all --ipc=host \
-v ${PROJ_DIR}:"/workspace/project" \
-v ${DATA_PATH}:"/workspace/datasets" \
-v ${OUTPUT_DIR}:/workspace/output_log \
-v ${MODEL_DIR}:/workspace/models \
-it --rm det2:my0