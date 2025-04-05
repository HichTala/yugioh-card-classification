import argparse
import json
import os
import requests
from tqdm import tqdm


class ZoulDownloader:

    def __init__(self):
        input_path = "./card_sets_augmented.json"
        self.output_path = "datasets/Zouloux/"

        with open(input_path, "rb") as f:
            self.card_sets = json.load(f)

        with open("card_sets_real.json", "rb") as f:
            self.output_dict = json.load(f)

        self.name = None
        self.card_set = None

    def download_file(self, url, destination):
        with requests.get(url) as response:
            if response.headers['content-type'] == 'image/jpeg':
                with open(destination, 'wb') as file:
                    file.write(response.content)
                if self.name not in self.output_dict:
                    self.output_dict[self.name] = [self.card_set]
                else:
                    self.output_dict[self.name].append(self.card_set)

    def download(self):
        for name, card_set_list in tqdm(self.card_sets.items(), desc="Dowloading Cards", colour='cyan'):
            self.name = name
            path = os.path.join(self.output_path, name.replace('/', ''))
            if not os.path.exists(path):
                os.makedirs(path)
            for card_set in card_set_list:
                self.card_set = card_set
                url = f"https://ygocard.s3.fr-par.scw.cloud/{card_set}.jpg"
                try:
                    self.download_file(url,
                                       os.path.join(path, f"{card_set}-{name.replace('/', '').split('-')[-1]}.jpg"))
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {url}: {e}")

                    with open("card_sets_real.json", "w") as json_file:
                        json.dump(self.output_dict, json_file)
                    continue


if __name__ == '__main__':
    downloader = ZoulDownloader()
    downloader.download()
