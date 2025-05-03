CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 \
    -m custom.train --cfg-path custom/projects/blip2/train/qm_retrieval.yaml