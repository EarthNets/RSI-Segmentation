_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/geonrw.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(num_classes=10),
    auxiliary_head=dict(num_classes=10),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
)
