from __future__ import print_function, division
import os
import cv2
cv2.setNumThreads(0)

import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]

def TrainDataAugmentation():
    transforms_train = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=256),
        albumentations.CenterCrop(height=224, width=224),
        albumentations.HorizontalFlip(),
        albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0.1, p=0.5),
        albumentations.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
        albumentations.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15),
        albumentations.Cutout(num_holes=2, max_h_size=8, max_w_size=8, fill_value=0, p=0.5),
        albumentations.Normalize(PRE__MEAN, PRE__STD),
        ToTensorV2(),
        ])
    return transforms_train

def TestDataAugmentation():
    transform_val = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=256),
        albumentations.CenterCrop(height=224, width=224),
        albumentations.Normalize(PRE__MEAN, PRE__STD),
        ToTensorV2(),
        ])
    return transform_val


class FacePAD_Train(Dataset):

    def __init__(self, image_info_lists, map_size=14):

        self.image_info_lists = pd.read_csv(image_info_lists)
        self.composed_transforms = TrainDataAugmentation()
        self.map_size = map_size

    def __len__(self):
        return len(self.image_info_lists)

    def __getitem__(self, index):

        image_path = self.image_info_lists.iloc[index, 0]
        label_str = self.image_info_lists.iloc[index, 1]

        image_x = cv2.imread(image_path)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)

        if label_str == 'attack':
            map_x = torch.zeros((self.map_size, self.map_size))
            spoofing_label = 0
        else:
            map_x = torch.ones((self.map_size, self.map_size))
            spoofing_label = 1

        image_x = self.composed_transforms(image=image_x)['image']

        return (image_x, int(spoofing_label), map_x)


class FacePAD_Val(Dataset):

    def __init__(self, image_info_lists, map_size=14):

        self.image_info_lists = pd.read_csv(image_info_lists)
        self.composed_transforms = TestDataAugmentation()
        self.map_size = map_size

    def __len__(self):
        return len(self.image_info_lists)

    def __getitem__(self, index):
        image_path = self.image_info_lists.iloc[index, 0]
        label_str = self.image_info_lists.iloc[index, 1]

        # obtain video name for computing a mean decision score of all frames of one video
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        video_id = image_name.rsplit('_', 1)[0]

        image_x = cv2.imread(image_path)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)

        if label_str == 'attack':
            map_x = torch.zeros((self.map_size, self.map_size))
            spoofing_label = 0
        else:
            map_x = torch.ones((self.map_size, self.map_size))
            spoofing_label = 1

        image_x = self.composed_transforms(image=image_x)['image']

        return (image_x, int(spoofing_label), map_x, video_id)
