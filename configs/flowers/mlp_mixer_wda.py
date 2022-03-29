_base_ = [
    '../_base_/models/mlp_mixer_base_patch16.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs32.py',
    '../_base_/default_runtime.py',
]
