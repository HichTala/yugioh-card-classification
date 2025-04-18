import argparse
import json
import os

import cv2
import requests
from PIL import Image
from tqdm import tqdm
from src.tools import art_cropper


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Dataset splitter', add_help=True)

    # data args
    parser.add_argument('--input_path', default='./cardinfo.json', type=str,
                        help="Path to training dataset's directory")
    parser.add_argument('--output_path', default='./output/', type=str,
                        help="Path to output formatted dataset's directory")
    return parser.parse_args()


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


def main(args):
    input_path = args.input_path
    output_path = args.output_path

    with open(input_path, "rb") as f:
        card_info = json.load(f)

    paths = {
        'spell': os.path.join(output_path, 'spell'),
        'trap': os.path.join(output_path, 'trap'),
        'effect': os.path.join(output_path, 'monster'),
        'normal': os.path.join(output_path, 'normal_monster'),
        'fusion': os.path.join(output_path, 'fusion'),
        'ritual': os.path.join(output_path, 'ritual')
    }

    for card in tqdm(card_info["data"], desc="Dowloading Cards", colour='cyan'):
        for i, images in enumerate(card["card_images"]):
            name = card["name"].replace(" ", "-") + "-" + str(i) + "-" + str(card["id"])
            # if card["frameType"] in paths.keys():
            path = os.path.join(output_path, name.replace('/', ''))
            if not os.path.exists(path):
                os.makedirs(path)
                file_path = os.path.join(path, str(images["id"])) + f'.{images["image_url_small"].split('.')[-1]}'
                download_file(images["image_url_small"], file_path)
                # image = Image.open(file_path)
                # image = art_cropper(image)
                # image.save(file_path)


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
