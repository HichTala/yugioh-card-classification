import os
import pickle

import cv2
import numpy as np
from PIL import Image
from torch import load
from torch.nn.functional import pairwise_distance

from siamese_network import SiameseNetwork
from tools import art_cropper, calculate_HOG_points, calculate_SIFT_score
from transformations import data_transforms, image_transforms, data_transforms_no_tensor, image_transforms_no_tensor


def predict(label, image_path):

    img0 = Image.open(image_path)
    input0 = image_transforms(img0)
    input0 = input0.cuda()
    input0 = input0.unsqueeze(dim=0)

    output0 = net.forward_once(input0)

    rank_list = []

    for key, value in feature_map.items():
        abs_path, output1 = value

        euclidean_distance = pairwise_distance(output0, output1, keepdim=True)

        rank_list.append((key, euclidean_distance.item(), abs_path))

    rank_list.sort(key=lambda x: x[1])
    rank_list_tot = np.array(rank_list)
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
    image_path_list = ['./queries/gg_0.jpg',
                       './queries/hl2_0.jpg',
                       './queries/isct_0.jpg',
                       './queries/sdc_0.jpg',
                       './queries/srl_0.jpg',
                       './queries/xyz.jpg',
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

    trained_model_path = "./models/res_epoch_60_012023.pth"
    feature_map_path = "./feature_maps/feature_map_60epochs.pkl"

    net = SiameseNetwork(None).cuda()
    net.load_state_dict(load(trained_model_path)['model_state_dict'])

    top_qualified = 300

    orb = cv2.ORB_create()

    with open(feature_map_path, 'rb') as f:
        feature_map = pickle.load(f)
    f.close()

    for index, image_path in enumerate(image_path_list):
        predict(labels[index], image_path)
