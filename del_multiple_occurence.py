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
import json

def main():
    input_path = "./card_sets_augmented.json"

    with open(input_path, "rb") as f:
        card_sets = json.load(f)

    with tqdm(total=13597, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk("datasets/cardmarket/"):
            for directory in dirs:
                pbar.update(1)
                for file in os.listdir(os.path.join(subdir, directory)):
                    abs_file_path = os.path.join(subdir, directory, file)
                    in_expansion = False
                    try:
                        for expansion in card_sets[directory]:
                            if expansion.startswith(file.split('-')[0]):
                                in_expansion = True
                                break

                        if not in_expansion:
                            shutil.move(abs_file_path, os.path.join("datasets/blacklist", file))
                    except KeyError:
                        continue






if __name__ == '__main__':
    main()
