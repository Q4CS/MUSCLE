import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def center_crop(img, crop_size=None):
    if crop_size is None:
        return img

    if len(crop_size) == 1:
        crop_w = crop_size[0]
        crop_h = crop_size[0]
    elif len(crop_size) == 2:
        crop_w = crop_size[0]
        crop_h = crop_size[1]
    else:
        return img

    if len(img.shape) == 3:
        w, h, c = img.shape  # [W, H, C]
    elif len(img.shape) == 2:
        w, h = img.shape  # [W, H]
    else:
        raise ValueError(f'image shape: {img.shape} not support!')
    w_start = int((w - crop_w) / 2)
    h_start = int((h - crop_h) / 2)

    if len(img.shape) == 3:
        out_img = img[w_start:(w_start + crop_w), h_start:(h_start + crop_h), :]
    elif len(img.shape) == 2:
        out_img = img[w_start:(w_start + crop_w), h_start:(h_start + crop_h)]
    else:
        raise ValueError(f'image shape: {img.shape} not support!')

    return out_img


def train_transform(p=1, img_size=(256, 256)):
    train_compose = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], always_apply=True),
            # Spatial-level transforms
            A.OneOf([
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.Transpose(p=0.2),
                A.RandomResizedCrop(height=img_size[1], width=img_size[0], scale=(0.8, 1.0), ratio=(0.8, 1.0), p=0.2),
            ], p=0.8),
            # Pixel-level transforms
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
                A.Blur(blur_limit=5, p=0.2),
                A.GaussNoise(p=0.2),
                A.ISONoise(p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 5), hole_width_range=(5,20), hole_height_range=(5,20), p=0.2),
            ], p=0.8),
            # A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(always_apply=True),  # apply `ToTensorV2` that converts a NumPy array to a PyTorch tensor
        ], p=p
    )
    return train_compose


def val_transform(p=1, img_size=(256, 256)):
    val_compose = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], always_apply=True),
            # A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(always_apply=True),  # apply `ToTensorV2` that converts a NumPy array to a PyTorch tensor
        ], p=p
    )
    return val_compose
