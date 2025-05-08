import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
from datasets.CheXpert.augmentation import val_transform, train_transform
from PIL import Image


base_dir = r'/home/datasets/CheXpert'


def split_list(full_list, ratio=None, shuffle=True):

    if ratio is None:
        ratio = [0.8, 0.2]

    if len(ratio) == 1:
        ratio.append(0.0)

    assert sum(ratio) <= 1, 'The ratio sum must be less than 1'

    sub_list_num = len(ratio)
    list_num = len(full_list)

    if shuffle:
        random.shuffle(full_list)

    sub_lists = []

    count = 0
    for i in range(sub_list_num):
        elem_num = round(list_num * ratio[i])
        if (i + 1) == sub_list_num:
            sub_lists.append(full_list[count:])
        else:
            sub_lists.append(full_list[count:(count + elem_num)])
        count = count + elem_num

    return sub_lists


def get_images_info_pd(base_dir, type_name='Pneumonia', split='train'):
    split_file_dir = os.path.join(base_dir, 'my_split', type_name)
    if os.path.exists(os.path.join(split_file_dir, 'split.csv')):
        split_info_pd = pd.read_csv(os.path.join(split_file_dir, 'split.csv'))
        images_info_pd = split_info_pd[split_info_pd['split']==split]
    else:
        all_info_pd = pd.read_csv(os.path.join(base_dir, 'train_cheXbert.csv'))
        target_type_pd = all_info_pd[(all_info_pd[type_name]==0) | (all_info_pd[type_name]==1)]
        path_list = target_type_pd['Path'].values.tolist()
        case_list = []
        for path_name in path_list:
            case_name = path_name.split('/')[2]
            if case_name not in case_list:
                case_list.append(case_name)
        sub_lists = split_list(full_list=case_list, ratio=[0.7, 0.1, 0.2], shuffle=True)

        new_target_type_pd = target_type_pd.copy()
        for index, row in target_type_pd.iterrows():
            now_case_name = row['Path'].split('/')[2]
            if now_case_name in sub_lists[0]:
                new_target_type_pd.loc[index, 'split'] = 'train'
            elif now_case_name in sub_lists[1]:
                new_target_type_pd.loc[index, 'split'] = 'val'
            elif now_case_name in sub_lists[2]:
                new_target_type_pd.loc[index, 'split'] = 'test'
            else:
                raise ValueError(f'case name not exist in train/val/test lists: {now_case_name}')
        if not os.path.exists(split_file_dir):
            os.makedirs(split_file_dir)
        new_target_type_pd.to_csv(os.path.join(split_file_dir, 'split.csv'), index=False)

        print(f"total: {len(new_target_type_pd)}, "
              f"train: {len(new_target_type_pd[new_target_type_pd['split']=='train'])}, "
              f"val: {len(new_target_type_pd[new_target_type_pd['split']=='val'])}, "
              f"test: {len(new_target_type_pd[new_target_type_pd['split']=='test'])}, ")

        images_info_pd = new_target_type_pd[new_target_type_pd['split']==split]

    return images_info_pd.reset_index(drop=True)


class CheXpert_Dataset(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            type_name='Pneumonia',
            image_size=(256, 256)
    ):
        super(CheXpert_Dataset, self).__init__()

        self.base_dir = base_dir
        self.split = split
        self.type_name = type_name
        self.image_size = image_size
        self.images_info_pd = get_images_info_pd(self.base_dir, type_name=self.type_name, split=self.split)

        if self.split == 'train':
            self.transform = train_transform(image_size=image_size)
        else:
            self.transform = val_transform(image_size=image_size)

        print(f'type_name: {self.type_name}, {self.split} set consists of {len(self.images_info_pd)} images')

    def __len__(self):
        return len(self.images_info_pd)

    def __getitem__(self, idx):
        image_info = self.images_info_pd.loc[idx]
        image_path = image_info['Path'].replace('CheXpert-v1.0', self.base_dir)
        image_path = image_path.replace('/train/', '/train_256x256/')
        image_label = image_info[self.type_name]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self.transform(image=image)
        image = transform['image']
        return {'image': image.float(),
                'cls_label': int(image_label),
                'image_name': image_path}
