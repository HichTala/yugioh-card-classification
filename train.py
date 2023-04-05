import torch.cuda
from torch import optim, save, load, cat, tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet101_Weights
from tqdm import tqdm

from card_dataset import CardDataset
from protonet import ProtoNet
from prototypical_loss import prototypical_loss as loss_fn
from sampler import EpisodicBatchSampler
from siamese_network import SiameseNetwork
from loss_functions import ContrastiveLoss
from transformations import data_transforms


class Config:
    training_dir = "./cardDatabaseFull/"

    train_number_epochs = 300
    resume_training = True

    lr = 0.0005

    n_way = 1000
    n_episodes = 22
    n_support = 5
    n_queries = 5


def data_initialization(n_way, n_episodes, n_support, n_queries):
    folder_dataset = datasets.ImageFolder(root=Config.training_dir)
    train_dataset = CardDataset(image_folder_dataset=folder_dataset, n_support=n_support, n_queries=n_queries,
                                transform=data_transforms)

    sampler = EpisodicBatchSampler(len(train_dataset), n_way, n_episodes)

    return DataLoader(train_dataset, sampler=sampler, num_workers=0)


def train(model, optimizer, train_dataloader, n_way, n_support, n_queries):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Start training")
    for epoch in range(Config.train_number_epochs):
        model.train()

        for batch in tqdm(train_dataloader, desc="Epoch {:d} train".format(epoch)):
            optimizer.zero_grad()

            assert batch['supports'].size(0) == batch['queries'].size(0)

            label = batch['label']
            supports = batch['supports']
            queries = batch['queries']

            label = label.view(n_way, 1, 1).expand(n_way, n_queries, 1).long()
            label = tensor(label, requires_grad=False)

            inputs = cat([
                supports.view(n_way * n_support, *supports.size()[2:]),
                supports.view(n_way * n_queries, *queries.size()[2:])
            ], dim=0)

            outputs = model(inputs)

            loss, results = loss_fn(outputs, label, n_way, n_support, n_queries)

            loss.backward()
            optimizer.step()

            savePath = './models/checkpoints/res_epoch_{}_012023.pth'.format(epoch)
            save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, savePath)

            epoch += 1

        savePath = './res-300-normalized.pth'
        save(model.state_dict(), savePath)


if __name__ == '__main__':
    train_dataloader = data_initialization(
        n_support=Config.n_support,
        n_queries=Config.n_queries,
        n_episodes=Config.n_episodes,
        n_way=Config.n_way
    )

    model = ProtoNet()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        n_way=Config.n_way,
        n_support=Config.n_support,
        n_queries=Config.n_queries
    )
