import albumentations as A
from albumentations.pytorch import ToTensorV2
from ResNet18_CBAM_MGA.core.config import CFG

train_transform_img_only = A.Compose([
    A.Resize(height=CFG.input_size[0], width=CFG.input_size[1]),
    A.HorizontalFlip(p=0.5),  # ✅ 이미지만 적용됨
    A.RandomBrightnessContrast(p=0.2),
    A.CoarseDropout(max_holes=1, max_height=40, max_width=40, fill_value=0, p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(*CFG.input_size),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
], additional_targets={"mask": "mask"})

# transform_no_rotate.py (마스크는 그대로)
tta_transforms = [
    A.Compose([
        A.Resize(*CFG.input_size),  # 반드시 추가
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]),
    A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(*CFG.input_size),  # 반드시 추가
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]),
    A.Compose([
        A.Rotate(limit=15, p=1.0),
        A.Resize(*CFG.input_size),  # 반드시 추가
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
]