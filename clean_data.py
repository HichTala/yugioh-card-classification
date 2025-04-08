import argparse
import os

from PIL import Image
from tqdm import tqdm

from src.tools import art_cropper
from src.transformations import image_transforms_no_tensor, train_data_transforms
import cv2


def main():
    dataset_size = 13160
    data_path = "datasets/ddraw"
    nb_image = 0

    with tqdm(total=dataset_size, desc="Augmenting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk(data_path):
            pbar.update(1)
            for file in files:
                nb_image += 1
                abs_file_path = os.path.join(subdir, file)
                img = cv2.imread(abs_file_path)
                img = cv2.resize(img, (89, 120), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(abs_file_path, img)
    print(nb_image, 'images')
                # try:
                #     img = Image.open(abs_file_path)
                # except IOError:
                #     print(iteration, subdir)
                #     iteration += 1


if __name__ == '__main__':
    main()
