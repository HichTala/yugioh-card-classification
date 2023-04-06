import numpy as np
import torch.cuda
from torch import optim, save, load, cat, tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from card_dataset import CardDataset
from protonet import ProtoNet
from prototypical_loss import prototypical_loss as loss_fn
from sampler import EpisodicBatchSampler
from transformations import data_transforms


class Config:
    training_dir = "./cardDatabaseFull/"

    train_number_epochs = 300
    resume_training = True

    lr = 0.0005

    n_way = 128  # Number of classes per episode
    n_episodes = 22  # Number of episodes
    n_support = 5  # Number of support examples per classes
    n_queries = 5  # Number of query examples per classes

    input_dim = 3
    hidden_dim = 64
    output_dim = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def data_initialization(n_way, n_episodes, n_support, n_queries):
    folder_dataset = datasets.ImageFolder(root=Config.training_dir)
    train_dataset = CardDataset(image_folder_dataset=folder_dataset, n_support=n_support, n_queries=n_queries,
                                transform=data_transforms)

    batch_sampler = EpisodicBatchSampler(len(train_dataset), n_way, n_episodes)

    return DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)


def train(model, optimizer, train_dataloader, n_way, n_support, n_queries):
    results_history = {'loss': [], 'acc': []}

    print("Start training")
    for epoch in range(Config.train_number_epochs):
        model.train()

        for batch in tqdm(train_dataloader, desc="Epoch {:d} train".format(epoch)):
            optimizer.zero_grad()

            assert batch['supports'].size(0) == batch['queries'].size(0)

            label = batch['label']
            supports = batch['supports'].to(Config.device)
            queries = batch['queries'].to(Config.device)

            label = label.view(n_way, 1, 1).expand(n_way, n_queries, 1).short().to(Config.device)

            inputs = cat([
                supports.view(n_way * n_support, *supports.size()[2:]),
                supports.view(n_way * n_queries, *queries.size()[2:])
            ], dim=0)

            outputs = model(inputs)

            loss, results = loss_fn(outputs, label, n_way, n_support, n_queries)

            loss.backward()
            optimizer.step()

            results_history['loss'].append(results['loss'])
            results_history['acc'].append(results['acc'])

        print("loss: {}, acc: {}".format(np.mean(results_history['loss']), np.mean(results_history['acc'].mean())))

        save_path = './models/checkpoints/proto_epoch_{}_050423.pth'.format(epoch)
        save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results_history
        }, save_path)

        epoch += 1

    save_path = './models/proto_050423.pth'
    save(model.state_dict(), save_path)


if __name__ == '__main__':
    train_dataloader = data_initialization(
        n_support=Config.n_support,
        n_queries=Config.n_queries,
        n_episodes=Config.n_episodes,
        n_way=Config.n_way
    )

    model = ProtoNet(
        input_dim=Config.input_dim,
        hidden_dim=Config.hidden_dim,
        output_dim=Config.output_dim
    ).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        n_way=Config.n_way,
        n_support=Config.n_support,
        n_queries=Config.n_queries
    )
