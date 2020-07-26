from pathlib import Path

import torch
from PIL import Image
import pandas as pd

import torchvision.transforms as transforms
import torch.utils.data as data

def make_datapath_list(root='./dataset', train_csv='train.csv', test_csv='test.csv'):
    train_csv_path = Path(root) / train_csv
    test_csv_path = Path(root) / test_csv

    train_img_list = list()
    train_lbl_list  = list()
    
    train_df = pd.read_csv(train_csv_path)

    for _, row in train_df.iterrows():
        img_filename = row['filename']
        img_path = Path(root) / 'images' / img_filename
        img_label = row['label']
        train_img_list.append(img_path)
        train_lbl_list.append(img_label)

    test_img_list = list()
    test_lbl_list  = list()
    
    test_df = pd.read_csv(test_csv_path)

    for _, row in test_df.iterrows():
        img_filename = row['filename']
        img_path = Path(root) / 'images' / img_filename
        img_label = row['label']
        test_img_list.append(img_path)
        test_lbl_list.append(img_label)

    return train_img_list, train_lbl_list, test_img_list, test_lbl_list


class DataTransforms():
    def __init__(self, img_size, color_mean, color_std, phase):

        if phase == 'train':
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(color_mean, color_std)
                             ])
        elif phase == 'test':
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(color_mean, color_std)
                             ])
        else:
            raise ValueError('custom datatransforms: this phase is not supported')

    def __call__(self, img):
        return self.transform(img)


class Dataset(data.Dataset):
    def __init__(self, img_list, lbl_list, transform):
        self.img_list = img_list
        self.lbl_list = lbl_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_filepath = self.img_list[index]
        img = Image.open(img_filepath)
        img = self.transform(img)

        lbl = self.lbl_list[index]

        return img, lbl