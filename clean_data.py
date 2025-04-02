import argparse
import os

from PIL import Image
from tqdm import tqdm

from src.tools import art_cropper
from src.transformations import image_transforms_no_tensor, train_data_transforms


def main():
    dataset_size = 13301
    data_path = "datasets/Zouloux"
    iteration = 0

    with tqdm(total=dataset_size, desc="Augmenting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                pbar.update(1)
                abs_file_path = os.path.join(subdir, file)

                try:
                    img = Image.open(abs_file_path)
                except IOError:
                    print(iteration, subdir)
                    iteration += 1


if __name__ == '__main__':
    main()
