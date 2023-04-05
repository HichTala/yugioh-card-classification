import cv2
import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data import Dataset


class CardDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):

        img_tuple = self.imageFolderDataset.imgs[index]

        img = Image.open(img_tuple[0])
        label = img_tuple[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
