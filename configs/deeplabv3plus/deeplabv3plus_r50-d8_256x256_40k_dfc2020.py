_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/dfc2020.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        type='ResNet',
        in_channels=13,
        ),
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
    pretrained=None,)
    
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

#evaluation = dict(interval=400, metric='mIoU', pre_eval=True)