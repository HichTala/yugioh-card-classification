import cv2
import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
import time


class CardDataset(Dataset):

    def __init__(self, image_folder_dataset, n_support, n_queries, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.n_support = n_support
        self.n_queries = n_queries
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = self.image_folder_dataset.imgs[index]

        img = Image.open(img_tuple[0])
        label = img_tuple[1]

        img = T.ToTensor()(img).unsqueeze(0)

        supports = [img]*self.n_support
        queries = [img]*self.n_queries

        supports = torch.cat(supports, dim=0)
        queries = torch.cat(queries, dim=0)

        if self.transform is not None:
            start = time.time()
            supports = self.transform(supports)
            queries = self.transform(queries)
            print(time.time()-start)

        return {'label': label, 'supports': supports, 'queries': queries}

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
