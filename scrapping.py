import argparse
import datetime
import json
import os
import time

import bs4
import requests
from seleniumbase import SB
from tqdm import tqdm

def main():
    input_path = "./card_sets.json"

    with open("cm_card_info.json", "rb") as f:
        output_dict = json.load(f)

    with open(input_path, "rb") as f:
        card_sets = json.load(f)

    names = [name for name in card_sets.keys() if name not in output_dict.keys()]

    with SB(uc=True, headless=True) as sb:
        for name in tqdm(names, desc="Scrapping cm", colour='cyan'):
            try:
                search_string = name[:-11].replace('"', '').replace("'", "")
                sb.open(f"https://www.cardmarket.com/en/YuGiOh/Products/Search?searchString={search_string}")
                sb.wait_for_element("select", timeout=10)
                page = sb.get_page_source()
                soup = bs4.BeautifulSoup(page, features="lxml")
                img_html_tags = soup.find('div', {"class": "table-body"}).find_all('span', {"class": "icon"})
                for i, img_html_tag in enumerate(img_html_tags):
                    img_soup = bs4.BeautifulSoup(img_html_tag.attrs['data-bs-original-title'], features="lxml")
                    if img_soup.find('img') is None:
                        url = soup.find_all('div', {"class": "slide"})[1].find('img').attrs['src']
                        if url.split('/')[-1] != 'cardImageNotAvailable.png':
                            if name not in output_dict:
                                output_dict[name] = [url]
                            else:
                                output_dict[name].append(url)
                        break
                    url = img_soup.find('img').attrs['src']
                    if url.split('/')[-1] != 'cardImageNotAvailable.png':
                        if name not in output_dict:
                            output_dict[name] = [url]
                        else:
                            output_dict[name].append(url)
                time.sleep(3)
            except Exception as e:
                print(e)
                print(name)
                with open("cm_card_info.json", "w") as json_file:
                    json.dump(output_dict, json_file)
                continue

    with open("cm_card_info.json", "w") as json_file:
        json.dump(output_dict, json_file)


if __name__ == '__main__':
    main()
