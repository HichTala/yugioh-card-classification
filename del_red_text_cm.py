import argparse
import os
import pathlib
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import math
import shutil


def main():
    ref = [36.4, 36.7, 239.8]
    with tqdm(total=13597, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk("datasets/cardmarket/"):
            for dir in dirs:
                pbar.update(1)
                for file in os.listdir(os.path.join(subdir, dir)):
                    abs_file_path = os.path.join(subdir, dir, file)
                    image = cv2.imread(abs_file_path)
                    image = cv2.resize(image, (268, 391), interpolation=cv2.INTER_LINEAR)
                    image = image[330:360, 60:200]
                    if math.dist(ref, [np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])]) <= 1:
                        shutil.move(abs_file_path, os.path.join("datasets/blacklist", file))




if __name__ == '__main__':
    main()
