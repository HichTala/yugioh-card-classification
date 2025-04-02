import argparse
import json
import os
import requests
from tqdm import tqdm


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


def main():
    input_path = "./card_sets.json"
    output_path = "datasets/Zouloux/"

    with open(input_path, "rb") as f:
        card_sets = json.load(f)

    for name, card_set in tqdm(card_sets.items(), desc="Dowloading Cards", colour='cyan'):
        path = os.path.join(output_path, name.replace('/', ''))
        if not os.path.exists(path):
            os.makedirs(path)
            url = f"https://ygocard.s3.fr-par.scw.cloud/{card_set}.jpg"
            download_file(url,  os.path.join(path, name.replace('/', '').split('-')[-1] + ".jpg"))


if __name__ == '__main__':
    main()
