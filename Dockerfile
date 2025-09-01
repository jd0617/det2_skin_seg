# Base image with PyTorch and CUDA support
# FROM nvcr.io/nvidia/pytorch:24.12-py3
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# RUN pip uninstall numpy -y
# RUN pip install numpy==1.24

# Install Python and essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev wget git && \
    rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
RUN mkdir -p /workspace/project
COPY . /workspace/project

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /workspace/project/requirements.txt

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN python -m pip install ultralytics --no-deps

RUN mkdir -p /workspace/output_log/
RUN mkdir -p /workspace/datasets/
RUN mkdir -p /workspace/models/

# Expose the default port for PyTorch distributed training
EXPOSE 29500

# Set the default command to an interactive shell
CMD ["bash"]