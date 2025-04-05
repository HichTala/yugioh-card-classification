import argparse
import json
import os
import requests
from tqdm import tqdm


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Dataset splitter', add_help=True)

    # data args
    parser.add_argument('--input_path', default='./cardinfo.json', type=str,
                        help="Path to training dataset's directory")
    # parser.add_argument('--input_path2', default='./cardinfo.json', type=str,
    #                     help="Path to training dataset's directory")
    return parser.parse_args()


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


def process(card_info, output_dict, set_date):
    card_types = ['spell', 'trap', 'effect', 'normal', 'fusion', 'ritual']
    for card in tqdm(card_info["data"], desc="Construction card sets equivalence", colour='cyan'):
        for i, images in enumerate(card["card_images"]):
            name = card["name"].replace(" ", "-") + "-" + str(i) + "-" + str(card["id"])
            # if card["frameType"] in card_types:
            if "card_sets" in card.keys() and card["frameType"] not in ["skill", "token"]:
                for card_set in card["card_sets"]:
                    try:
                        if name not in output_dict:
                            output_dict[name] = [card_set['set_code'].replace("EN", "FR")]
                        else:
                            output_dict[name].append(card_set['set_code'].replace("EN", "FR"))
                    except KeyError:
                        pass
                # rl_date = None
                #
                #     if card_set['set_code'].split("-")[0] in set_date.keys():
                #         rl_date = set_date[card_set['set_code'].split("-")[0]]
                #
                #         break
                # if rl_date is not None:
                #     for card_set in card["card_sets"]:
                #         try:
                #             if rl_date < set_date[card_set['set_code'].split("-")[0]]:
                #                 rl_date = set_date[card_set['set_code'].split("-")[0]]
                #                 output_dict[name] = card_set['set_code'].replace("EN", "FR")
                #         except KeyError:
                #             pass


def main(args):
    input_path = args.input_path
    # input_path2 = args.input_path2

    output_dict = {}
    set_date = {}

    with open(input_path, "rb") as f:
        card_info = json.load(f)

    # with open(input_path2, "rb") as f:
    #     card_info2 = json.load(f)

    with open("cardsets.json", "rb") as f:
        cardsets = json.load(f)

    for set in cardsets:
        if set["set_code"] not in set_date:
            if "tcg_date" in set.keys() and not set["set_code"].startswith("OP"):
                set_date[set["set_code"]] = set["tcg_date"]

    process(card_info, output_dict, set_date)
    # process(card_info2, output_dict, set_date)

    with open("card_sets_augmented.json", "w") as json_file:
        json.dump(output_dict, json_file)


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
