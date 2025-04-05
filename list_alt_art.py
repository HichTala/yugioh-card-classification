import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    with tqdm(total=13597, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk("datasets/ygoprodeck/"):
            for dir in dirs:
                pbar.update(1)
                try:
                    if dir.split("-")[-2] == "1":
                        print(dir)
                except IndexError:
                    breakpoint()
                    continue


if __name__ == '__main__':
    main()
