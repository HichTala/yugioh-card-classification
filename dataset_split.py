import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Dataset splitter', add_help=True)

    # data args
    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")
    parser.add_argument('--dataset_size', default=10856, type=int,
                        help="Number of cards in total in the dataset (default: 10856)")
    parser.add_argument('--output_path', default='./cardDatabaseFormatted/', type=str,
                        help="Path to output formatted dataset's directory")

    return parser.parse_args()


def main(args):
    dataset_size = args.dataset_size
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    paths = {
        'spell': os.path.join(output_path, 'spell'),
        'trap': os.path.join(output_path, 'trap'),
        'monster': os.path.join(output_path, 'monster'),
        'normal_monster': os.path.join(output_path, 'normal_monster'),
        'link': os.path.join(output_path, 'link'),
        'xyz': os.path.join(output_path, 'xyz'),
        'fusion': os.path.join(output_path, 'fusion'),
        'pendulum': os.path.join(output_path, 'pendulum'),
        'syncro': os.path.join(output_path, 'syncro'),
        'ritual': os.path.join(output_path, 'ritual'),
        'other': os.path.join(output_path, 'other')
    }

    colors = {
        'spell': np.array([26, 143, 135]),
        'trap': np.array([162, 46, 117]),
        'monster': np.array([174, 101, 46]),
        'normal_monster': np.array([192, 147, 90]),
        'link': np.array([20, 86, 144]),
        'xyz': np.array([31, 31, 31]),
        'syncro': np.array([226, 225, 223]),
        'fusion': np.array([135, 62, 154]),
        'pendulum': np.array([174, 103, 47]),
        'ritual': np.array([82, 121, 188])
    }

    card_types = ['spell', 'trap', 'monster', 'normal_monster', 'link', 'xyz', 'syncro', 'fusion', 'pendulum', 'ritual']

    count = 0

    with tqdm(total=dataset_size, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk(args.data_path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                    count += 1
                    pbar.update(1)
                    abs_file_path = os.path.join(subdir, file)

                    img = np.array(Image.open(abs_file_path))

                    counts = [
                        np.abs(img[16, 16] - colors[card_type]).sum()
                        for card_type in card_types
                    ]

                    argmin = np.argmin(counts)

                    if counts[argmin] == 0:
                        if not os.path.exists(os.path.join(paths[card_types[argmin]], os.path.basename(subdir))):
                            os.makedirs(os.path.join(paths[card_types[argmin]], os.path.basename(subdir)))

                        shutil.copy(abs_file_path,
                                    os.path.join(os.path.join(paths[card_types[argmin]], os.path.basename(subdir)),
                                                 file))
                    else:
                        if not os.path.exists(paths['other']):
                            os.makedirs(paths['other'])

                        shutil.copy(abs_file_path, os.path.join(paths['other'], file))


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
