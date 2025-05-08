import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def train_transform(image_size=(256, 256)):
    transform = A.Compose([
        A.Resize(width=image_size[0], height=image_size[1], always_apply=True),
        A.RandomRotate90(p=0.1),
        A.Flip(p=0.1),
        A.Transpose(p=0.1),
        A.GaussNoise(var_limit=(0.0, 0.001), p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.1),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        # ToFloat(),
        ToTensorV2(),  # apply `ToTensorV2` that converts a NumPy array to a PyTorch tensor
    ])
    return transform


def val_transform(image_size=(256, 256)):
    transform = A.Compose([
        A.Resize(width=image_size[0], height=image_size[1], always_apply=True),
        # ToFloat(),
        ToTensorV2(),  # apply `ToTensorV2` that converts a NumPy array to a PyTorch tensor
    ])
    return transform
