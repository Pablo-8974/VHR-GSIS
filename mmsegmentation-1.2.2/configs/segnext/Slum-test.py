_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_UIS.py',
    '../_base_/datasets/Slum-test.py'
]
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (416, 416)
data_preprocessor = dict(
    type='SegDataPreProcessor',

    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[97.041084397831, 97.870065454629, 95.633268563905],
    std=[49.599847618176, 42.17493460328, 43.138548238378],

    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(256, 256),
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
train_dataloader = dict(batch_size=64)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(   # 微调将这里的学习率改小了，注意调整
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),   # 0.00006   0.00001
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(                                                          # 这里的end也改小了，数据量太小
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),    # 1500
    dict(
        type='PolyLR',
        power=1.0,
        begin=100,    # 这里也相应改小了1500  100
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

val_evaluator = dict(
    type='IoUMetric',
    output_dir='/intelnvme04/jiang.mingyu/slum/mmsegmentation-1.2.2/segnext-uis/vis',  # 验证结果保存路径
    format_only=False,
    keep_results=True)  # 设置为True会保存可视化结果

load_from = '/intelnvme04/jiang.mingyu/slum/mmsegmentation-1.2.2/slum-segnext/Mumbai/best_mIoU_epoch_0.pth'
