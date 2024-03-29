import argparse
import os
import pickle

import numpy as np
from PIL import Image
from torch import load
from torch.cuda import is_available
from tqdm import tqdm

from src.resnet import ResNet
from src.transformations import final_data_transforms


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Card classification feature map parser', add_help=True)

    # data args
    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")
    parser.add_argument('--limit', default=20, type=int,
                        help="Number of cards per partitions")
    parser.add_argument('--dataset_size', default=4864, type=int,
                        help="Number of cards in total in the dataset (default: 4864)")

    # model args
    parser.add_argument('model_path', default='', type=str,
                        help="Path to trained model's checkpoint")
    parser.add_argument('--device', type=str, default=None,
                        help="device to use for training (default: cuda if available cpu otherwise)")

    # option args
    parser.add_argument('--partition', action='store_false',
                        help="create feature map partitions (default: True)")
    parser.add_argument('--merge', action='store_false',
                        help="merge feature map partitions (default: True)")

    return parser.parse_args()


def prototype_partition(model, limit, dataset_size):
    device = 'cuda' if is_available() else 'cpu'

    feature_map = {}

    count = 0
    limit_count = 0
    with tqdm(total=dataset_size, desc="Saving pickles", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk(args.data_path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):

                    count += 1
                    pbar.update(1)
                    abs_file_path = os.path.join(subdir, file)

                    img = Image.open(abs_file_path)
                    inputs = final_data_transforms(img)
                    inputs = inputs.to(device)
                    inputs = inputs.unsqueeze(dim=0)

                    outputs = model(inputs)

                    dir_name = subdir.split('/')[-1]

                    feature_map[dir_name] = (abs_file_path, outputs)

                    if count % limit == 0:
                        save_path = './prototypes/prototypes_partition/prototype_{}.pkl'.format(limit_count)

                        with open(save_path, "wb") as f:
                            pickle.dump(feature_map, f)
                        f.close()

                        feature_map = {}
                        limit_count += 1

    save_path = './prototypes/prototypes_partition/prototype_{}.pkl'.format(limit_count + 1)

    with open(save_path, "wb") as f:
        pickle.dump(feature_map, f)
    f.close()

    print("All cards have been saved in {} partitions".format(limit_count))
    return limit_count


def merge_prototype(partition_number):
    feature_map_dir = 'prototypes/prototypes_partition/'
    feature_map = {}

    with tqdm(total=partition_number, desc="Merging pickles", colour='cyan') as pbar:
        for files in os.listdir(feature_map_dir):
            abs_path = feature_map_dir + files

            with open(abs_path, "rb") as f:
                tmp = pickle.load(f)
            f.close()

            feature_map.update(tmp)
            del tmp

            savePath = './prototypes/prototype.pkl'
            with open(savePath, "wb") as f:
                pickle.dump(feature_map, f)
            f.close()
            pbar.update(1)
    print("{} partitions have been merged !".format(partition_number))


def main(args):
    if args.device is None:
        device = 'cuda' if is_available() else 'cpu'
    else:
        device = args.device

    model = ResNet().to(device)
    model.load_state_dict(load(args.model_path)['model_state_dict'])

    partition_number = np.ceil(args.dataset_size / args.limit)

    if args.partition:
        partition_number = prototype_partition(model=model, limit=args.limit, dataset_size=args.dataset_size)
    if args.merge:
        merge_prototype(partition_number=partition_number)


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
