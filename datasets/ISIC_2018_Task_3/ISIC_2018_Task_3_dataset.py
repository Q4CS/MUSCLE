import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datasets.ISIC_2018_Task_3.augmentation import train_transform, val_transform, center_crop

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


base_dir = r'/home/datasets/ISIC_2018_Task_3'


def read_imgs_path_list(img_dir, label_csv_path):
    reindex_columns_list = ['image', 'NV', 'DF', 'BKL', 'VASC', 'AKIEC','BCC', 'MEL']

    label_csv_pd = pd.read_csv(label_csv_path, header=0)
    label_csv_pd = label_csv_pd.reindex(columns=reindex_columns_list)
    label_argmax = np.argmax(label_csv_pd[['NV', 'DF', 'BKL', 'VASC', 'AKIEC','BCC', 'MEL']].values, axis=1)
    label_csv_pd['argmax'] = label_argmax

    imgs_path_list = []
    for idx, row in label_csv_pd.iterrows():
        img_path = os.path.join(img_dir, row['image'] + '.jpg')
        if not os.path.exists(img_path):
            raise ValueError(f'image does not exist: {img_path}')
        imgs_path_list.append([img_path, row['argmax']])

    return imgs_path_list


class ISIC2018Task3_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            img_size=(256, 256),
    ):
        super(ISIC2018Task3_DataSets, self).__init__()

        self.base_dir = base_dir
        self.split = split
        if self.split == 'train':
            self.img_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_Training_Input')
            self.label_csv_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Training_GroundTruth',
                                               'ISIC2018_Task3_Training_GroundTruth.csv')
            self.transform = train_transform(img_size=img_size)
        elif self.split == 'val':
            self.img_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_Validation_Input')
            self.label_csv_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Validation_GroundTruth',
                                               'ISIC2018_Task3_Validation_GroundTruth.csv')
            self.transform = val_transform(img_size=img_size)
        elif self.split == 'test':
            self.img_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_Test_Input')
            self.label_csv_path = os.path.join(self.base_dir, 'ISIC2018_Task3_Test_GroundTruth',
                                               'ISIC2018_Task3_Test_GroundTruth.csv')
            self.transform = val_transform(img_size=img_size)
        else:
            raise ValueError(f"The split ({split}) must be between 'train', 'val' or 'test'")

        self.images_path_list = read_imgs_path_list(self.img_dir, self.label_csv_path)

        print(f'split: {self.split}, total {len(self.images_path_list)} samples')

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        image_path = self.images_path_list[idx][0]
        cls_label = self.images_path_list[idx][1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape
        crop_size = image_shape[0] if image_shape[0] <= image_shape[1] else image_shape[1]  # min edge

        image = center_crop(image, crop_size=[crop_size, crop_size])

        transform = self.transform(image=image)
        image = transform['image']

        return {'image': image.float(),
                'cls_label': cls_label,
                'image_name': os.path.basename(image_path)}
