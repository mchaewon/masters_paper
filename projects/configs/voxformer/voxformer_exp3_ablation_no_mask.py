_base_ = ['./voxformer_exp3_3d.py']
work_dir = 'results/ablation_no_mask'
model = dict(lambda_obj=0.0)
