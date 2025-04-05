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
    i = 0
    last = ""
    for subdir, dirs, files in os.walk("datasets/cardmarket/"):
        for dir1 in dirs:
            for dir2 in dirs:
                if "-".join(dir2.split("-")[:-2]).startswith("-".join(dir1.split("-")[:-2])) and "-".join(dir1.split("-")[:-2]) != "-".join(dir2.split("-")[:-2]):
                    if last != "-".join(dir1.split("-")[:-2]):
                        print(i, "-".join(dir1.split("-")[:-2]), "-".join(dir2.split("-")[:-2]))
                        last = "-".join(dir1.split("-")[:-2])
                        i += 1





if __name__ == '__main__':
    main()
