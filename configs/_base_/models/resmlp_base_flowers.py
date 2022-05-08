# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResMlp',
        #TODO: the following initialization parameters may need changes
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='LinearClsHead',
        num_classes=17,
        # x shape after patch embed = (batch size, 196, embed dim)
        in_channels=196,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)
