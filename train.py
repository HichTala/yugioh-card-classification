import argparse

import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import wandb
from torch import save, load, cat, no_grad
from torch.cuda import is_available
from torch.utils.data import DataLoader, Subset
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
    parser.add_argument('--lr', default=1e-5, type=float,
                        help="learning rate (default: 0.00001)")
    parser.add_argument('--device', type=str, default=None,
                        help="device to use for training (default: cuda if available cpu otherwise)")

    # hyperparameter args
    parser.add_argument('--n_way', type=int, default=32,
                        help="Number of classes per episodes (default: 32")
    parser.add_argument('--n_episodes', type=int, default=341,
                        help="Number of episodes (default: 341)")
    parser.add_argument('--n_partition', type=int, default=128,
                        help="Number of classes per partitions (default: 128")
    parser.add_argument('--n_supports', type=int, default=5,
                        help="Number of support examples per classes (default: 5)")
    parser.add_argument('--n_queries', type=int, default=5,
                        help="Number of query examples per classes (default: 5)")
    parser.add_argument('--n_classes', type=int, default=4864,
                        help="Number of classes in the dataset(default: 4864)")

    # resume training
    parser.add_argument('--resume', default=None, type=str,
                        help="Path to the checkpoint to resume from (default: None)")
    parser.add_argument('--no_sampler', action='store_true',
                        help="Do not use a sampler for training (default: False)")

    return parser.parse_args()


def data_initialization(training_dir, n_classes, n_supports, n_queries):
    folder_dataset = datasets.ImageFolder(root=training_dir)
    train_dataset = CardDataset(
        image_folder_dataset=folder_dataset,
        n_supports=n_supports,
        n_queries=n_queries,
        transform=train_data_transforms
    )
    train_dataset = Subset(train_dataset, range(n_classes))

    return train_dataset


def proto_preprocess(model, train_dataset, n_partition, n_supports, device):
    preprocess_loader = DataLoader(train_dataset, batch_size=n_partition)
    prototypes = []

    for i, batch in enumerate(tqdm(preprocess_loader, desc="\033[1mPreprocessing\033[0m", colour='green')):
        supports = batch['supports'].to(device)

        # TODO: Find a way to avoid error when n_classes isn't a multiple of n_partition
        inputs = cat([
            supports.view(n_partition * n_supports, *supports.size()[2:]),
        ], dim=0)

        outputs = model(inputs)
        dim = outputs.size(-1)
        prototypes.append(outputs.view(n_partition, n_supports, dim).mean(1))
    return cat(prototypes)


def train(
        model,
        optimizer,
        results_history,
        train_dataloader,
        prototypes,
        start_epochs,
        end_epochs,
        n_supports,
        n_queries,
        n_way,
        device
):
    writer = SummaryWriter('outputs/runs')
    n_iter = 0

    print("Start training")
    model.train()
    for epoch in range(start_epochs, end_epochs):
        for batch in tqdm(train_dataloader, desc="\033[1mEpoch {:d}\033[0m train".format(epoch), colour='cyan'):
            optimizer.zero_grad()

            assert batch['supports'].size(0) == batch['queries'].size(0)

            labels = batch['label']
            supports = batch['supports'].to(device)
            queries = batch['queries'].to(device)

            inputs = cat([
                supports.view(n_way * n_supports, *supports.size()[2:]),
                queries.view(n_way * n_queries, *queries.size()[2:])
            ], dim=0)

            outputs = model(inputs)

            loss, results = loss_fn(
                outputs=outputs,
                labels=labels,
                prototypes=prototypes,
                n_way=n_way,
                n_supports=n_supports,
                n_queries=n_queries,
                device=device
            )
            loss.backward()

            optimizer.step()

            results_history['loss'].append(results['loss'])
            results_history['acc'].append(results['acc'])

            writer.add_scalar('Train/Loss', results['loss'], n_iter)
            writer.add_scalar('Train/Accuracy', results['acc'], n_iter)
            n_iter += 1

            wandb.log({
                'loss': results['loss'],
                'acc': results['acc'],
            })

        print("\033[1m\033[96mloss\033[0m: {}, \033[1m\033[96macc\033[0m: {}".format(
            np.mean(results_history['loss']),
            np.mean(results_history['acc'])
        ))

        wandb.log({
            'mean-loss': np.mean(results_history['loss']),
            'mean-acc': np.mean(results_history['acc']),
        })

        save_path = './outputs/checkpoints/proto_epoch_{}.pth'.format(epoch)
        save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results_history,
            'prototypes': prototypes,
            'epochs': epoch
        }, save_path)

    wandb.finish()

    save_path = './outputs/proto.pth'
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
            "n_way": args.n_way
        }
    )

    train_dataset = data_initialization(
        training_dir=args.data_path,
        n_classes=args.n_classes,
        n_supports=args.n_supports,
        n_queries=args.n_queries,
    )

    if args.no_sampler:
        train_dataloader = DataLoader(train_dataset, batch_size=args.n_way, num_workers=0)
    else:
        batch_sampler = EpisodicBatchSampler(
            n_classes=args.n_classes,
            n_episodes=args.n_episodes,
            n_way=args.n_way
        )
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)

    model = ResNet().to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    results_history = {'loss': [], 'acc': []}
    epochs = -1

    with no_grad():
        prototypes = proto_preprocess(
            model=model,
            train_dataset=train_dataset,
            n_partition=args.n_partition,
            n_supports=args.n_supports,
            device=device
        )

    if args.resume is not None:
        state_dict = load(args.resume)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        epochs = state_dict['epochs']
        prototypes = state_dict['prototypes']

    train(
        model=model,
        optimizer=optimizer,
        results_history=results_history,
        train_dataloader=train_dataloader,
        prototypes=prototypes,
        start_epochs=epochs + 1,
        end_epochs=args.epochs,
        n_supports=args.n_supports,
        n_queries=args.n_queries,
        n_way=args.n_way,
        device=device
    )


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
