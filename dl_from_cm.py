import argparse
import json
import os
import requests
from tqdm import tqdm


class CMDownloader:

    def __init__(self):
        input_path = "cm_card_info_new.json"
        self.output_path = "./datasets/cardmarket/"

        with open(input_path, "rb") as f:
            self.cm_card = json.load(f)

        with open("cm_card_sets_real.json", "rb") as f:
            self.output_dict = json.load(f)

        self.name = None
        self.cm_url = None

    def download_file(self, url, destination):
        with requests.get(url, headers={'referer': "https://www.cardmarket.com/"}) as response:
            # if response.headers['content-type'] == 'image/jpeg':
            with open(destination, 'wb') as file:
                file.write(response.content)
            if self.name not in self.output_dict:
                self.output_dict[self.name] = [self.cm_url]
            else:
                self.output_dict[self.name].append(self.cm_url)

    def download(self):
        for name, cm_url_list in tqdm(self.cm_card.items(), desc="Dowloading Cards", colour='cyan'):
            self.name = name
            path = os.path.join(self.output_path, name.replace('/', ''))
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                if len(os.listdir(path)) == len(cm_url_list):
                    continue
            for url in cm_url_list:
                self.cm_url = url
                card_set = url.split('/')[-3]
                try:
                    self.download_file(url,
                                       os.path.join(path,
                                                    f"{card_set}-{url.split('/')[-2]}.{url.split('.')[-1]}"))
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {url}: {e}")

                    with open("cm_card_sets_real.json", "w") as json_file:
                        json.dump(self.output_dict, json_file)
                    continue


if __name__ == '__main__':
    downloader = CMDownloader()
    downloader.download()
