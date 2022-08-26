_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/loveda_uda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        num_classes=7
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=7))
