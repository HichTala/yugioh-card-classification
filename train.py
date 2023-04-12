import argparse
import pickle

import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import wandb
from torch import save, load, cat
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from src.card_dataset import CardDataset
from src.resnet import ResNet
from src.prototypical_loss import prototypical_loss as loss_fn
from src.sampler import EpisodicBatchSampler
from src.transformations import train_data_transforms


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Card classification training parser', add_help=True)

    # data args
    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")

    # train args
    parser.add_argument('--epochs', default=300, type=int,
                        help="Number of epochs to train (default: 300)")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="learning rate (default: 0.0001)")
    parser.add_argument('--device', type=str, default=None,
                        help="device to use for training (default: cuda if available cpu otherwise)")

    # hyperparameter args
    parser.add_argument('--n_way', type=int, default=8,
                        help="Number of classes per partitions (default: 8")
    # parser.add_argument('--n_episodes', type=int, default=2,
    #                     help="Number of episodes (default: 2)")
    parser.add_argument('--n_supports', type=int, default=5,
                        help="Number of support examples per classes (default: 5)")
    parser.add_argument('--n_queries', type=int, default=5,
                        help="Number of query examples per classes (default: 5)")

    # resume training
    parser.add_argument('--resume', default=None, type=str,
                        help="Path to the checkpoint to resume from (default: None)")

    return parser.parse_args()


def data_initialization(training_dir, n_way, n_supports, n_queries):
    folder_dataset = datasets.ImageFolder(root=training_dir)
    train_dataset = CardDataset(image_folder_dataset=folder_dataset, n_supports=n_supports, n_queries=n_queries,
                                transform=train_data_transforms)

    batch_sampler = EpisodicBatchSampler(len(train_dataset), n_way)

    return DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)


def train(
        model,
        optimizer,
        scheduler,
        results_history,
        train_dataloader,
        epochs,
        n_way,
        n_supports,
        n_queries,
        device
):
    writer = SummaryWriter('models/runs')
    n_iter = 0

    print("Start training")
    model.train()
    for epoch in range(epochs):

        for i, batch in enumerate(
                tqdm(train_dataloader, desc="\033[1mEpoch {:d}\033[0m train".format(epoch), colour='cyan')):
            optimizer.zero_grad()

            assert batch['supports'].size(0) == batch['queries'].size(0)

            supports = batch['supports'].to(device)
            queries = batch['queries'].to(device)

            inputs = cat([
                supports.view(n_way * n_supports, *supports.size()[2:]),
                queries.view(n_way * n_queries, *queries.size()[2:])
            ], dim=0)

            outputs = model(inputs)

            supports_path = './models/pickles/supports/pickle_{}'.format(i)
            with open(supports_path, "wb") as f:
                pickle.dump(outputs[:n_way * n_supports], f)
            f.close()

            queries_path = './models/pickles/queries/pickle_{}'.format(i)
            with open(queries_path, "wb") as f:
                pickle.dump(outputs[n_way * n_queries:], f)
            f.close()

            del outputs

        loss, results = loss_fn(n_way, n_supports, n_queries)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(results['loss'])

        results_history['loss'].append(results['loss'])
        results_history['acc'].append(results['acc'])

        writer.add_scalar('Train/Loss', results['loss'], n_iter)
        writer.add_scalar('Train/Accuracy', results['acc'], n_iter)
        n_iter += 1

        wandb.log({
            'loss': results['loss'],
            'acc': results['acc'],
            'mean-loss': np.mean(results_history['loss']),
            'mean-acc': np.mean(results_history['acc']),
        })

        print("\033[1m\033[96mloss\033[0m: {}, \033[1m\033[96macc\033[0m: {}".format(
            results['loss'],
            results['acc']
        ))

        save_path = './models/checkpoints/proto_epoch_{}.pth'.format(epoch)
        save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results_history
        }, save_path)

    wandb.finish()

    save_path = './models/proto.pth'
    save(model.state_dict(), save_path)


def main(args):
    if args.device is None:
        device = 'cuda' if is_available() else 'cpu'
    else:
        device = args.device

    wandb.init(
        project="ygo-card-classification",
        config={
            "learning_rate": args.lr,
            "architecture": 'Resnet128',
            "scheduler": ''
        }
    )

    train_dataloader = data_initialization(
        training_dir=args.data_path,
        n_supports=args.n_supports,
        n_queries=args.n_queries,
        n_way=args.n_way
    )

    model = ResNet().to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min')
    results_history = {'loss': [], 'acc': []}

    if args.resume is not None:
        state_dict = load(args.resume)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        results_history = state_dict['results']

    train(
        model=model,
        optimizer=optimizer,
        scheduler=None,
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
