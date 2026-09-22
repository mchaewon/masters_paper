_base_ = ['./voxformer_exp3_3d.py']
work_dir = 'results/ablation_no_depth'
model = dict(lambda_ego=0.0)
