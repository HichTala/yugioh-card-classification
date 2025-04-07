import argparse
import datetime
import json
import os
import time

import bs4
import requests
from seleniumbase import SB
from tqdm import tqdm

def format_search(search_string):
    search_string = search_string.replace('"', '').replace("'", "")
    search_string = search_string.replace('---', '-').replace("-=-", "-")
    search_string = search_string.replace(',', '').replace(".", "")
    search_string = search_string.replace('!', '').replace('?', '')
    search_string = search_string.replace('%', '').replace("#", "")
    search_string = search_string.replace('@', '').replace("/", "")
    search_string = search_string.replace('★', '').replace("☆", "")
    search_string = search_string.replace('&-', '').replace(':', '')
    search_string = search_string.replace('(', '').replace(')', '')

    return search_string

def main():
    input_path = "./card_sets_augmented.json"
    output_path = "cm_card_info_new.json"
    wrong_format_path = "wrong_format.json"
    right_format_path = "right_format.json"

    with open(output_path, "rb") as f:
        output_dict = json.load(f)

    with open(wrong_format_path, "rb") as f:
        wrong_format_dict = json.load(f)

    with open(right_format_path, "rb") as f:
        right_format_dict = json.load(f)

    with open(input_path, "rb") as f:
        card_sets = json.load(f)

    names = [name for name in card_sets.keys() if name not in output_dict.keys() and name]
    print(names)

    with SB(uc=True, headless=True) as sb:
        for name in tqdm(names, desc="Scrapping cm", colour='cyan'):
            try:
                search_string = "-".join(name.split("-")[:-2])
                if name in right_format_dict:
                    search_string = right_format_dict[name]
                search_string = format_search(search_string)
                sb.open(f"https://www.cardmarket.com/en/YuGiOh/Cards/{search_string}/Versions")
                # sb.wait_for_element("select", timeout=10)
                time.sleep(3)
                if sb.is_element_visible('button[class="btn btn-outline-primary"]'):
                    sb.click('button[class="btn btn-outline-primary"]')
                    time.sleep(3)

                    scroll_pause = 1
                    scroll_step = 1000
                    total_height = sb.driver.execute_script("return document.body.scrollHeight")
                    current_position = 0
                    while current_position < total_height:
                        sb.driver.execute_script(f"window.scrollTo(0, {current_position});")
                        time.sleep(scroll_pause)
                        current_position += scroll_step
                        total_height = sb.driver.execute_script("return document.body.scrollHeight")
                    sb.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)

                page = sb.get_page_source()
                soup = bs4.BeautifulSoup(page, features="lxml")
                img_html_tags = soup.find_all('div', {"class": "image card-image is-yugioh is-sharp"})
                for i, tag in enumerate(img_html_tags):
                    url = tag.find('img').attrs['src']
                    if url.split('/')[-1] != 'cardImageNotAvailable.png':
                        if name not in output_dict:
                            output_dict[name] = [url]
                        else:
                            output_dict[name].append(url)
                time.sleep(3)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                print(name)
                with open(output_path, "w") as json_file:
                    json.dump(output_dict, json_file)
                continue

    print("Saving data...")
    with open(output_path, "w") as json_file:
        json.dump(output_dict, json_file)
    with open(wrong_format_path, "w") as json_file:
        json.dump(wrong_format_dict, json_file)


if __name__ == '__main__':
    main()
