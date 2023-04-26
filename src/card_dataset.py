import torchvision.transforms as T
from PIL import Image
from torch import cat
from torch.utils.data import Dataset

from src.tools import art_cropper


class CardDataset(Dataset):

    def __init__(self, image_folder_dataset, n_supports, n_queries, transform):
        self.image_folder_dataset = image_folder_dataset
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.transform = transform

    def __getitem__(self, index):
        img_tuple = self.image_folder_dataset.imgs[index]

        img = Image.open(img_tuple[0])
        img = art_cropper(img)
        label = img_tuple[1]

        img = T.ToTensor()(img).unsqueeze(0)

        supports = [self.transform(img) for _ in range(self.n_supports)]
        queries = [self.transform(img) for _ in range(self.n_queries)]

        if supports:
            supports = cat(supports, dim=0)
        if queries:
            queries = cat(queries, dim=0)

        return {'label': label, 'supports': supports, 'queries': queries}  # {'supports': supports, 'queries': queries}

    def __len__(self):
        return len(self.image_folder_dataset.imgs)
