import os
import shutil

from tqdm import tqdm


def main():
    dataset1 = "datasets/Zouloux"
    dataset2 = "/home/hicham/Documents/draw2-training/datasets/ddraw"

    with tqdm(total=13597, desc="Formatting Dataset", colour='cyan') as pbar:
        for subdir, dirs, files in os.walk(dataset1):
            for directory in dirs:
                pbar.update(1)
                for file in os.listdir(os.path.join(subdir, directory)):
                    output = os.path.join(dataset2, "-".join(directory.split("-")[:-2]) + "-" + directory.split("-")[-1], file)
                    shutil.copy(os.path.join(subdir, directory, file), output)

if __name__ == '__main__':
    main()