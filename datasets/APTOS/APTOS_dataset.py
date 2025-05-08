import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datasets.APTOS.augmentation import train_transform, val_transform, center_crop
from utils import split_list

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

base_dir = r'/home/datasets/aptos2019-blindness-detection'


def APTOS_dataset_division(base_dir, shuffle=False):
    division_file_name = os.path.join(base_dir, 'APTOS_dataset_division.csv')
    train_csv_path = os.path.join(base_dir, 'train.csv')
    if os.path.exists(division_file_name):
        all_img_info_pd = pd.read_csv(division_file_name, header=0)
        train_img_info_pd = all_img_info_pd[all_img_info_pd['division'] == 'train']
        val_img_info_pd = all_img_info_pd[all_img_info_pd['division'] == 'val']
        test_img_info_pd = all_img_info_pd[all_img_info_pd['division'] == 'test']

        train_list = []
        for idx, row in train_img_info_pd.iterrows():
            train_list.append([row['image_name'], row['cls_label']])

        val_list = []
        for idx, row in val_img_info_pd.iterrows():
            val_list.append([row['image_name'], row['cls_label']])

        test_list = []
        for idx, row in test_img_info_pd.iterrows():
            test_list.append([row['image_name'], row['cls_label']])

    else:
        train_list = []
        val_list = []
        test_list = []

        for i in range(5):
            train_csv_pd = pd.read_csv(train_csv_path, header=0)
            temp_cls_pd = train_csv_pd[train_csv_pd['diagnosis'] == i]
            cls_all_list = []
            for idx, row in temp_cls_pd.iterrows():
                temp_img_name = row['id_code'] + '.png'
                temp_img_path = os.path.join(base_dir, 'train_images', temp_img_name)
                if not os.path.exists(temp_img_path):
                    raise ValueError(f'{temp_img_path} does not exist!')

                cls_all_list.append([temp_img_name, row['diagnosis']])

            sub_lists = split_list(cls_all_list, ratio=[0.7, 0.1, 0.2], shuffle=shuffle)
            train_list.extend(sub_lists[0])
            val_list.extend(sub_lists[1])
            test_list.extend(sub_lists[2])

        # save to csv
        all_img_name = []
        all_img_cls = []
        all_img_division = []
        for info in train_list:
            all_img_name.append(info[0])
            all_img_cls.append(info[1])
            all_img_division.append('train')
        for info in val_list:
            all_img_name.append(info[0])
            all_img_cls.append(info[1])
            all_img_division.append('val')
        for info in test_list:
            all_img_name.append(info[0])
            all_img_cls.append(info[1])
            all_img_division.append('test')
        all_img_info_pd = pd.DataFrame.from_dict({'image_name': all_img_name,
                                                  'cls_label': all_img_cls,
                                                  'division': all_img_division})
        all_img_info_pd.to_csv(division_file_name, index=False)

    return {'train_list': train_list,
            'val_list': val_list,
            'test_list': test_list}


def resize_img_keep_ratio(cv2_img, target_size):
    old_size = cv2_img.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(cv2_img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


class APTOS_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            img_size=(256, 256)
    ):
        super(APTOS_DataSets, self).__init__()

        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.dataset_division = APTOS_dataset_division(self.base_dir, shuffle=False)

        if split == 'train':
            self.images_mask_info_list = self.dataset_division['train_list']
            self.transform = train_transform(img_size=img_size)
        elif split == 'val':
            self.images_mask_info_list = self.dataset_division['val_list']
            self.transform = val_transform(img_size=img_size)
        elif split == 'test':
            self.images_mask_info_list = self.dataset_division['test_list']
            self.transform = val_transform(img_size=img_size)
        else:
            raise ValueError(f"The split ({split}) must be between 'train', 'val' or 'test'")

        print(f'split: {self.split}, total {len(self.images_mask_info_list)} samples')

    def __len__(self):
        return len(self.images_mask_info_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_dir, 'train_images', self.images_mask_info_list[idx][0])
        cls_label = self.images_mask_info_list[idx][1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = resize_img_keep_ratio(image, self.img_size)

        image_shape = image.shape
        crop_size = image_shape[0] if image_shape[0] <= image_shape[1] else image_shape[1]  # min edge
        image = center_crop(image, crop_size=[crop_size, crop_size])

        transform = self.transform(image=image)
        image = transform['image']

        return {'image': image.float(),
                'cls_label': cls_label,
                'image_name': os.path.basename(image_path)}
