import cv2
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from sm_dataset import SiameseNetworkDataset


class Config:
    # training_dir = "./data/cards_old/training/"
    # testing_dir = "./data/cards_old/testing/"

    # training_dir = "./data/cards/training/"
    # testing_dir = "./data/cards/testing/"

    # training_dir = "./cardDatabase/"
    # testing_dir = "./cardDatabase/"

    training_dir = "./cardDatabaseFull/"
    testing_dir = "./cardDatabaseFull/"

    # testing_dir = "./data/cards_old/testing/"

    # train_batch_size = 24
    train_batch_size = 24 * 2

    # train_batch_size = 8
    train_number_epochs = 300


if __name__ == '__main__':
    data_transforms = transforms.Compose([
                        transforms.Resize(50),
                        transforms.Resize((300, 200)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
    ])

    folder_dataset = datasets.ImageFolder(root=Config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=data_transforms)

    dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=8)

    dataiter = iter(dataloader)


    raise breakpoint()
