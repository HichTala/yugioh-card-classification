import argparse
import os
import shutil

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


def regulate_dir(subdir, dir):
    target_path = "./datasets/cardmarket/"

    list_alt_dir = []
    hash_list = []
    i = 0
    name = dir.split("-")
    name[-2] = str(i)
    name = '-'.join(name)
    while True:
        path = os.path.join(subdir, name)
        if os.path.exists(path):
            list_alt_dir.append(name)
            for file in os.listdir(path):
                abs_file_path = os.path.join(path, file)
                img = Image.open(abs_file_path).resize((268, 391), Image.Resampling.LANCZOS)
                hash_list.append((imagehash.average_hash(img, 32), imagehash.phash(img, 32)))
            i += 1
            name = name.split("-")
            name[-2] = str(i)
            name = '-'.join(name)
        else:
            break
    if os.path.exists(os.path.join(target_path, dir)):
        for file in os.listdir(os.path.join(target_path, dir)):
            abs_file_path = os.path.join(target_path, dir, file)
            img = Image.open(abs_file_path).resize((268, 391), Image.Resampling.LANCZOS)
            hash = imagehash.average_hash(img, 32)
            phash = imagehash.phash(img, 32)
            hamming_distance = [(hash - h[0]) + (phash - h[1]) for h in hash_list]
            try:
                minimum = min(hamming_distance)
            except ValueError:
                breakpoint()
            nb_art = hamming_distance.index(minimum)
            if minimum > 400:
                print(dir, file, minimum, nb_art)
            for i, alt_dir in enumerate(list_alt_dir):
                if i != nb_art:
                    os.remove(os.path.join(target_path, alt_dir, file))


def main():
    with tqdm(total=13597, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk("datasets/ygoprodeck/"):
            for dir in dirs:
                pbar.update(1)
                if dir.split("-")[-2] == "1":
                    regulate_dir(subdir, dir)


if __name__ == '__main__':
    main()
