import json


def main():
    input_path = "./card_sets_augmented.json"

    with open("cm_card_info1.json", "rb") as f:
        output_dict1 = json.load(f)

    with open("cm_card_info2.json", "rb") as f:
        output_dict2 = json.load(f)

    with open("cm_card_info3.json", "rb") as f:
        output_dict3 = json.load(f)

    with open("cm_card_info4.json", "rb") as f:
        output_dict4 = json.load(f)

    with open("cm_card_info.json", "rb") as f:
        output_dict = json.load(f)

    output_dict.update(output_dict1)
    output_dict.update(output_dict2)
    output_dict.update(output_dict3)
    output_dict.update(output_dict4)


    with open("cm_card_info_new.json", "w") as json_file:
        json.dump(output_dict, json_file)

if __name__ == '__main__':
    main()