# dataset settings
dataset_type = "ADE20KDataset"
data_root = "data/ade/ADEChallengeData2016"
image_normalization_config = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
max_ratio = 4
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", img_scale=(512 * max_ratio, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **image_normalization_config),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["image", "gt_semantic_seg"]),
]
validation_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **image_normalization_config),
            dict(type="ImageToTensor", keys=["image"]),
            dict(type="Collect", keys=["image"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **image_normalization_config),
            dict(type="ImageToTensor", keys=["image"]),
            dict(type="Collect", keys=["image"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/training",
        ann_dir="annotations/training",
        pipeline=train_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["images/training", "images/validation"],
        ann_dir=["annotations/training", "annotations/validation"],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation",
        ann_dir="annotations/validation",
        pipeline=validation_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="testing",
        pipeline=test_pipeline,
    ),
)
