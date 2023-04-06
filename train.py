import argparse

import numpy as np
from torch import optim, save, load, cat, arange
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from card_dataset import CardDataset
from protonet import ProtoNet
from prototypical_loss import prototypical_loss as loss_fn
from sampler import EpisodicBatchSampler
from transformations import train_data_transforms


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Card classification training parser', add_help=True)

    # data args
    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")

    # train args
    parser.add_argument('--epochs', default=300, type=int,
                        help="Number of epochs to train (default: 300)")
    parser.add_argument('--lr', default=5e-4, type=float,
                        help="learning rate (default: 0.001)")
    parser.add_argument('--device', type=str, default=None,
                        help="device to use for training (default: cuda if available cpu otherwise)")

    # model args
    parser.add_argument('--input_dim', type=int, default=3,
                        help="input image number of channel (default: 3)")
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="hidden layer dimensions (default: 64)")
    parser.add_argument('--output_dim', type=int, default=64,
                        help="model output dimensions")

    # hyperparameter args
    parser.add_argument('--n_way', type=int, default=64,
                        help="Number of classes per episode")
    parser.add_argument('--n_episodes', type=int, default=390,
                        help="Number of episodes")
    parser.add_argument('--n_supports', type=int, default=5,
                        help="Number of support examples per classes")
    parser.add_argument('--n_queries', type=int, default=5,
                        help="Number of query examples per classes")

    # resume training
    parser.add_argument('--resume', default=None, type=str,
                        help="Path to the checkpoint to resume from (default: None)")

    return parser.parse_args()


def data_initialization(training_dir, n_way, n_episodes, n_supports, n_queries):
    folder_dataset = datasets.ImageFolder(root=training_dir)
    train_dataset = CardDataset(image_folder_dataset=folder_dataset, n_supports=n_supports, n_queries=n_queries,
                                transform=train_data_transforms)

    batch_sampler = EpisodicBatchSampler(len(train_dataset), n_way, n_episodes)

    return DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)


def train(model, optimizer, results_history, train_dataloader, epochs, n_way, n_supports, n_queries, device):
    print("Start training")
    for epoch in range(epochs):
        model.train()

        for batch in tqdm(train_dataloader, desc="Epoch {:d} train".format(epoch), colour='cyan'):
            optimizer.zero_grad()

            assert batch['supports'].size(0) == batch['queries'].size(0)

            supports = batch['supports'].to(device)
            queries = batch['queries'].to(device)

            label = arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_queries, 1).long().to(device)

            inputs = cat([
                supports.view(n_way * n_supports, *supports.size()[2:]),
                supports.view(n_way * n_queries, *queries.size()[2:])
            ], dim=0)

            outputs = model(inputs)

            loss, results = loss_fn(outputs, label, n_way, n_supports, n_queries)

            loss.backward()
            optimizer.step()

            results_history['loss'].append(results['loss'])
            results_history['acc'].append(results['acc'])

        print("loss: {}, acc: {}".format(np.mean(results_history['loss']), np.mean(results_history['acc'])))

        save_path = './models/checkpoints/proto_epoch_{}.pth'.format(epoch)
        save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results_history
        }, save_path)

        epoch += 1

    save_path = './models/proto.pth'
    save(model.state_dict(), save_path)


def main(args):
    if args.device is None:
        device = 'cuda' if is_available() else 'cpu'
    else:
        device = args.device

    train_dataloader = data_initialization(
        training_dir=args.data_path,
        n_supports=args.n_supports,
        n_queries=args.n_queries,
        n_episodes=args.n_episodes,
        n_way=args.n_way
    )

    model = ProtoNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    results_history = {'loss': [], 'acc': []}

    if args.resume is not None:
        state_dict = load(args.resume)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        results_history = state_dict['results']

    train(
        model=model,
        optimizer=optimizer,
        results_history=results_history,
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        n_way=args.n_way,
        n_supports=args.n_supports,
        n_queries=args.n_queries,
        device=device
    )


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
