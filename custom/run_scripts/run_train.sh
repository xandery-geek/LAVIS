gpu_num=$1
gpu_id=$2

export CUDA_VISIBLE_DEVICES=$gpu_id

torchrun --nproc_per_node=$gpu_num --master_port=9527 \
    -m custom.train --cfg-path custom/projects/blip2/train/qm_retrieval.yaml