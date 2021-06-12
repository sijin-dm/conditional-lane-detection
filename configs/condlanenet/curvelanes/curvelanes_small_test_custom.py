"""
    config file of the small version of CondLaneNet for curvelanes
"""
# global settings
dataset_type = 'CurvelanesDataset'
# Please change to your images' path. The default suffix is '.jpg'.
# Modify `test_suffix` in this config if yours is different.
data_root = "imx490" 
mask_down_scale = 8
hm_down_scale = 16
mask_size = (1, 40, 100)
line_width = 3
radius = 4
lane_nms_thr = -1
num_lane_classes = 1
batch_size = 1
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
img_scale = (800, 320)
train_cfg = dict(out_scale=mask_down_scale)
test_cfg = dict(out_scale=mask_down_scale)

# model settings
model = dict(
    type='CurvelanesRnn',
    pretrained='torchvision://resnet18',
    train_cfg=train_cfg,
    test_cfg=test_cfg,
    num_classes=num_lane_classes,
    backbone=dict(
        type='ResNet',
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='TransConvFPN',
        in_channels=[128, 256, 64],
        out_channels=64,
        num_outs=3,
        trans_idx=-1,
        trans_cfg=dict(
            in_dim=512,
            attn_in_dims=[512, 64],
            attn_out_dims=[64, 64],
            strides = [1, 1],
            ratios=[4, 4],
            pos_shape=(batch_size, 10, 25),
        ),
        ),
    head=dict(
        type='CondLaneRNNHead',
        heads=dict(hm=num_lane_classes),
        in_channels=(64, ),
        num_classes=num_lane_classes,
        head_channels=64,
        head_layers=1,
        disable_coords=False,
        branch_channels=64,
        branch_out_channels=64,
        reg_branch_channels=64,
        branch_num_conv=1,
        hm_idx=1,
        mask_idx=0,
        compute_locations_pre=True,
        zero_hidden_state=True,
        ct_head=dict(
            heads=dict(hm=1, params=128),
            channels_in=64,
            final_kernel=1,
            head_conv=128),
        location_configs=dict(size=(batch_size, 1, 40, 100), device='cuda:0')),
)

val_al_pipeline = [
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]


val_pipeline = [
    dict(type='albumentation', pipelines=val_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectRNNLanes',
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'filename', 'sub_img_name', 'gt_masks', 'mask_shape', 'hm_shape',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points', 'crop_shape', 'crop_offset'
        ]),
]

data = dict(
    samples_per_gpu=
    batch_size,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=data_root ,
        data_list=data_root ,
        test_suffix='.jpg', # image suffix.
        pipeline=val_pipeline,
        test_mode=True,
    ))

