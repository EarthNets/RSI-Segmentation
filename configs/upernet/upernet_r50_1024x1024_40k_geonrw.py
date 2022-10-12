_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/geonrw.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=10),
    auxiliary_head=dict(num_classes=10),
    pretrained='pretrained_weights/resnet50_v1c-2cccc1ad.pth'
    )
