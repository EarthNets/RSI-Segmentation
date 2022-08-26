_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/loveda_u2r.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=7), auxiliary_head=dict(num_classes=7))

evaluation = dict(interval=8000, metric='mIoU', pre_eval=False)
workflow = [('train', 1)]

optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
