import cv2
import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data import Dataset


class CardDataset(Dataset):

    def __init__(self, image_folder_dataset, n_support, n_query, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.n_support = n_support
        self.n_query = n_query
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = self.image_folder_dataset.imgs[index]

        img = Image.open(img_tuple[0])
        label = img_tuple[1]

        if self.transform is not None:
            supports = [self.transform(img) for _ in range(self.n_support)]
            queries = [self.transform(img) for _ in range(self.n_query)]
        else:
            supports = img
            queries = img

        return {'label': label, 'supports': supports, 'queries': queries}

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
