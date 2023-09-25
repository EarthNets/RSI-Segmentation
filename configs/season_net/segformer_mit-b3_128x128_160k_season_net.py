_base_ = ['../_base_/models/segformer_mit-b0.py', '../_base_/datasets/season_net.py',
          '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 18, 3]),
    decode_head=dict(num_classes=33, in_channels=[64, 128, 320, 512]))

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

expr_name = 'segformer_mit-b3_128x128_160k_season_net'

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

