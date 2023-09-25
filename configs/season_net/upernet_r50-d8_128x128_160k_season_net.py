_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/season_net.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(num_classes=33),
    auxiliary_head=dict(num_classes=33),
)

evaluation = dict(interval=16000, metric='mIoU', pre_eval=False)
workflow = [('train', 1)]

expr_name = 'upernet_r50-d8_128x128_160k_season_net'

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
