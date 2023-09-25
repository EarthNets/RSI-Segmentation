_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/season_net.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa

model = dict(
    decode_head=dict(num_classes=33,
                     in_channels=768,
                     c1_in_channels=96,
                     c1_channels=48),
    auxiliary_head=dict(num_classes=33, in_channels=384),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-5,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


evaluation = dict(interval=16000, metric='mIoU', pre_eval=False)
workflow = [('train', 1)]

expr_name = 'deeplabv3plus_convnext_t-d8_128x128_160k_2e-4_season_net'

init_kwargs = dict(
    project='rsi_segmentation',
    entity='tum-tanmlh',
    name=expr_name,
    resume='never'
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbHookSeg',
        #      init_kwargs=init_kwargs,
        #      interval=201),
        dict(type='MMSegWandbHook',
             init_kwargs=init_kwargs,
             interval=501,
             num_eval_images=20,
             BGR2RGB=True),
        # dict(type='PseudoLabelingHook',
        #      log_dir='work_dirs/pseudo_labels/deeplabv3plus_r50-d8_512x512_80k_loveda_r2u',
        #      interval=1),
    ])
