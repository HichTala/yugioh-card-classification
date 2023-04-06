import argparse
import os
import pickle

from PIL import Image
from torch import load

from transformations import data_transforms

def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Card classification feature map parser', add_help=True)

if __name__ == '__main__':

    dataset_dir = "./cardDatabaseFull/"
    trained_model_path = "./models/res_epoch_60_012023.pth"

    print("Loading model: " + trained_model_path)
    net = SiameseNetwork(None).cuda()
    net.load_state_dict(load(trained_model_path)['model_state_dict'])
    print("Model loaded successfully")

    feature_map = {}

    count = 0
    limit = 20
    limit_count = 0
    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):

                count += 1
                abs_file_path = os.path.join(subdir, file)

                img0 = Image.open(abs_file_path)
                input0 = data_transforms(img0)
                input0 = input0.cuda()
                input0 = input0.unsqueeze(dim=0)

                output0 = net.forward_once(input0)

                dir_name = subdir.split('/')[-1]

                feature_map[dir_name] = (abs_file_path, output0)

                if count % limit == 0:
                    save_path = './feature_maps/feature_maps_partition/feature_map_{}.pkl'.format(limit_count)

                    print("Saving pickle...")
                    with open(save_path, "wb") as f:
                        pickle.dump(feature_map, f)
                    f.close()

                    print("{} Cards has been saved to feature_map_{}.pkl".format(len(feature_map), limit_count))
                    feature_map = {}
                    limit_count += 1

    save_path = './feature_map/feature_map_{}.pkl'.format(limit_count)

    print("Saving pickle...")
    with open(save_path, "wb") as f:
        pickle.dump(feature_map, f)

    print("{} Cards has been saved to feature_map_{}".format(len(feature_map), limit_count))
