import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from torch import load
from torch.cuda import is_available

from src.resnet import ResNet
from src.tools import art_cropper, calculate_HOG_points, calculate_SIFT_score
from src.transformations import image_transforms, image_transforms_no_tensor


class Config:
    image_path_list = ['./queries/gg_0.jpg',
                       './queries/hl2_0.jpg',
                       './queries/isct_0.jpg',
                       './queries/sdc_0.jpg',
                       './queries/srl_0.jpg',
                       './queries/251194600.jpg',
                       './queries/rh_0.jpg',
                       './queries/pog_0.jpg']

    labels = ['Giant-Germ-0-95178994',
              'Harpie-Lady-2-0-27927359',
              'Flying-Kamakiri-1-0-84834865',
              'Chaos-Sorcerer-0-9596126',
              'Swords-of-Revealing-Light-0-72302403',
              'YZTank-Dragon-0-25119460',
              'Harpie-Queen-0-75064463',
              'Pot-of-Greed-0-55144522']

    trained_model_path = "./models/lr=0.00001/proto_epoch_122.pth"
    feature_map_path = "../feature_maps/lr=0.00001/feature_map_122eps.pkl"


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def predict(label, image_path):
    img0 = Image.open(image_path)
    input0 = image_transforms(img0)
    input0 = input0.cuda()
    input0 = input0.unsqueeze(dim=0)

    output0 = model(input0)

    rank_list = []
    raise breakpoint()

    for key, value in feature_map.items():
        abs_path, output1 = value

        euclidean_distance = euclidean_dist(output0, output1)

        rank_list.append((key, euclidean_distance.item(), abs_path))

    rank_list.sort(key=lambda x: x[1])
    rank_list_tot = np.array(rank_list)
    print(rank_list_tot[0, 0:2])
    rank_list = np.array(rank_list[:top_qualified])
    rank_list_dict = {key: (value, i) for i, (key, value, _) in enumerate(rank_list)}
    rank_list_dict_tot = {key: (value, i) for i, (key, value, _) in enumerate(rank_list_tot)}
    print(label, rank_list_dict_tot[label])

    if label in rank_list_dict.keys():

        final_rank_list = []

        for card in rank_list:
            img0_cropped = art_cropper(image_transforms_no_tensor(img0)).convert('RGB')

            img1 = Image.open(card[2])
            img1_cropped = art_cropper(image_transforms_no_tensor(img1)).convert('RGB')

            img0_cropped = np.array(img0_cropped)[:, :, ::-1].copy()
            img1_cropped = np.array(img1_cropped)[:, :, ::-1].copy()

            num_points = calculate_HOG_points(orb, img0_cropped, img1_cropped)
            final_score = calculate_SIFT_score(float(card[1]), num_points)
            # final_score = -num_points
            final_rank_list.append((card[0], card[1], final_score, card[2]))

        # print()
        # print()

        final_rank_list.sort(key=lambda x: x[2])
        final_rank_list = np.array(final_rank_list)
        # print(final_rank_list[:, 0:3][:10])

        final_rank_list_dict = {key: (final_value, i) for i, (key, value, final_value, _) in enumerate(final_rank_list)}

        # print()
        # print()
        print(label, final_rank_list_dict[label])


if __name__ == '__main__':
    device = 'cuda' if is_available() else 'cpu'

    model = ResNet().to(device)
    model.load_state_dict(load(Config.trained_model_path)['model_state_dict'])
    model.eval()

    top_qualified = 300

    orb = cv2.ORB_create()

    with open(Config.feature_map_path, 'rb') as f:
        feature_map = pickle.load(f)
    f.close()

    for index, image_path in enumerate(Config.image_path_list):
        predict(Config.labels[index], image_path)
