optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.001,eps=1e-8)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
# # learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
