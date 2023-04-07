import argparse
import os
import pickle

from PIL import Image
from torch import load
from torch.cuda import is_available
from tqdm import tqdm

from resnet import ResNet
from transformations import final_data_transforms


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Card classification feature map parser', add_help=True)

    # data args
    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")

    # model args
    parser.add_argument('--input_dim', type=int, default=3,
                        help="input image number of channel (default: 3)")
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="hidden layer dimensions (default: 64)")
    parser.add_argument('--output_dim', type=int, default=64,
                        help="model output dimensions")
    parser.add_argument('model_path', default='./cardDatabaseFull/', type=str,
                        help="Path to trained model's checkpoint")

    return parser.parse_args()


def feature_map_partition(model):
    device = 'cuda' if is_available() else 'cpu'

    feature_map = {}

    count = 0
    limit = 20
    limit_count = 0
    for subdir, dirs, files in tqdm(os.walk(args.data_path), desc="Saving pickles:", colour='cyan'):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):

                count += 1
                abs_file_path = os.path.join(subdir, file)

                img = Image.open(abs_file_path)
                inputs = final_data_transforms(img)
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(dim=0)

                outputs = model(inputs)

                dir_name = subdir.split('/')[-1]

                feature_map[dir_name] = (abs_file_path, outputs)

                if count % limit == 0:
                    save_path = './feature_maps/feature_maps_partition/feature_map_{}.pkl'.format(limit_count)

                    with open(save_path, "wb") as f:
                        pickle.dump(feature_map, f)
                    f.close()

                    feature_map = {}
                    limit_count += 1

    save_path = './feature_map/feature_map_{}.pkl'.format(limit_count)

    with open(save_path, "wb") as f:
        pickle.dump(feature_map, f)

    print("{} Cards has been saved to feature_map_{}".format(len(feature_map), limit_count))


def merge_feature_map():
    feature_map_dir = './feature_maps/feature_maps_partition/'
    feature_map = {}

    for i, files in enumerate(tqdm(os.listdir(feature_map_dir), desc="Merging pickles:", colour='cyan')):
        abs_path = feature_map_dir + files

        with open(abs_path, "rb") as f:
            tmp = pickle.load(f)
        f.close()

        feature_map.update(tmp)
        del tmp

        if i % 100 == 0:
            print("{} Pickle files merged already !".format(i))

        print('{} Cards saved'.format(len(feature_map)))

        print("Saving feature map")
        savePath = './feature_map.pkl'
        with open(savePath, "wb") as f:
            pickle.dump(feature_map, f)
        f.close()
        print('Feature map saved!')


def main(args):
    device = 'cuda' if is_available() else 'cpu'

    model = ResNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    model.load_state_dict(load(args.model_path)['model_state_dict'])

    feature_map_partition(model=model)
    merge_feature_map()


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
