import os
import random
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datasets.Kvasir.augmentation import train_transform, val_transform, center_crop

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


base_dir = r'/home/datasets/Kvasir/kvasir-dataset-v2'


classes_dict = {'esophagitis': 0,
                'dyed-lifted-polyps': 1,
                'dyed-resection-margins': 2,
                'normal-cecum': 3,
                'normal-pylorus': 4,
                'normal-z-line': 5,
                'polyps': 6,
                'ulcerative-colitis': 7}


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


def read_imgs_labels_info_list(base_dir, split):
    csv_name_list = ['img_name', 'class_name', 'label', 'division']
    imgs_labels_info_list = []
    train_val_test_info_csv = 'train_val_test_info.csv'
    train_val_test_info_csv_path = os.path.join(base_dir, train_val_test_info_csv)
    if os.path.exists(train_val_test_info_csv_path):
        train_val_test_info_pd = pd.read_csv(train_val_test_info_csv_path, header=0)
    else:
        for class_name, label in classes_dict.items():
            class_dir = os.path.join(base_dir, class_name)
            imgs_name_list = os.listdir(class_dir)
            sub_lists = split_list(full_list=imgs_name_list, ratio=[0.7, 0.1, 0.2], shuffle=False)
            for img_name in sub_lists[0]:
                imgs_labels_info_list.append([img_name, class_name, label, 'train'])
            for img_name in sub_lists[1]:
                imgs_labels_info_list.append([img_name, class_name, label, 'val'])
            for img_name in sub_lists[2]:
                imgs_labels_info_list.append([img_name, class_name, label, 'test'])

        train_val_test_info_pd = pd.DataFrame(columns=csv_name_list, data=imgs_labels_info_list)
        train_val_test_info_pd.to_csv(train_val_test_info_csv_path, index=False)

    neee_split_pd = train_val_test_info_pd[train_val_test_info_pd['division'] == split]
    return neee_split_pd.values.tolist()



class KvasirV2_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            img_size=(256, 256),
    ):
        super(KvasirV2_DataSets, self).__init__()

        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size

        if self.split == 'train':
            self.transform = train_transform(img_size=img_size)
        elif self.split == 'val':
            self.transform = val_transform(img_size=img_size)
        elif self.split == 'test':
            self.transform = val_transform(img_size=img_size)
        else:
            raise ValueError(f"The split ({split}) must be between 'train', 'val' or 'test'")

        self.images_labels_info_list = read_imgs_labels_info_list(base_dir=self.base_dir, split=self.split)

        print(f'split: {self.split}, total {len(self.images_labels_info_list)} samples')

    def __len__(self):
        return len(self.images_labels_info_list)

    def __getitem__(self, idx):
        image_name = self.images_labels_info_list[idx][0]
        class_name = self.images_labels_info_list[idx][1]
        cls_label = self.images_labels_info_list[idx][2]

        image_path = os.path.join(self.base_dir, class_name, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        transform = self.transform(image=image)
        image = transform['image']

        return {'image': image.float(),
                'cls_label': cls_label,
                'image_name': image_name}
