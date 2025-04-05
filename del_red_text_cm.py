import argparse
import os
import pathlib
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    with tqdm(total=13597, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk("datasets/Zouloux/"):
            for dir in dirs:
                pbar.update(1)
                try:
                    if len(os.listdir(os.path.join(subdir, dir))) == 0:
                        pathlib.Path.rmdir(os.path.join(subdir, dir))
                except IndexError:
                    breakpoint()
                    continue


if __name__ == '__main__':
    main()
