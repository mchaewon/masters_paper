#!/usr/bin/env bash


python tools/train.py projects/configs/voxformer/voxformer_exp3_ablation_no_depth.py --work-dir results/ablation_no_depth

python tools/train.py projects/configs/voxformer/voxformer_exp3_ablation_no_mask.py --work-dir results/ablation_no_mask