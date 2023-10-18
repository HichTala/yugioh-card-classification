import argparse
import os

from PIL import Image
from tqdm import tqdm

from src.tools import art_cropper
from src.transformations import image_transforms_no_tensor, train_data_transforms


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Dataset splitter testing parser', add_help=True)

    # data args
    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")
    parser.add_argument('--dataset_size', default=10856, type=int,
                        help="Number of cards in total in the dataset (default: 10856)")

    # iterations

    parser.add_argument('--iter', default=100, type=int,
                        help="Number of augmented images (default: 100)")

    return parser.parse_args()


def main(args):
    dataset_size = args.dataset_size

    with tqdm(total=dataset_size, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk(args.data_path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                    pbar.update(1)
                    abs_file_path = os.path.join(subdir, file)

                    img = Image.open(abs_file_path)

                    img = art_cropper(img)
                    img = image_transforms_no_tensor(img)
                    img.save(abs_file_path)

                    for k in range(args.iter):
                        split = file.split('.')
                        file = split[0] + f"_{k}." + split[1]

                        img = train_data_transforms(img)
                        img.save(os.path.join(subdir, file))


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
