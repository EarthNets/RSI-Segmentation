_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/geonrw.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# vit-small / deit-small
checkpoint_file = 'pretrained_weights/modified_deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth'
model = dict(
    pretrained='pretrained_weights/modified_deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth',
    backbone=dict(
        num_heads=6,
        embed_dims=384,
        drop_path_rate=0.1,
        ),
    decode_head=dict(num_classes=10, in_channels=[384, 384, 384, 384]),
    neck=None,
    auxiliary_head=dict(num_classes=10, in_channels=384))



# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
