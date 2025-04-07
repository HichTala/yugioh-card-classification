import json


def main():
    input_path = "./card_sets_augmented.json"

    with open("card_sets1.json", "rb") as f:
        output_dict1 = json.load(f)

    with open("card_sets2.json", "rb") as f:
        output_dict2 = json.load(f)

    with open("card_sets3.json", "rb") as f:
        output_dict3 = json.load(f)

    with open("card_sets4.json", "rb") as f:
        output_dict4 = json.load(f)

    with open("cm_card_info.json", "rb") as f:
        output_dict = json.load(f)

    with open(input_path, "rb") as f:
        card_sets = json.load(f)

    names = [name for name in card_sets.keys() if name not in output_dict.keys()]

    for i, (key, values) in enumerate(card_sets.items()):
        if key in names:
            if i % 4 == 0:
                output_dict1[key] = values
            if i % 4 == 1:
                output_dict2[key] = values
            if i % 4 == 2:
                output_dict3[key] = values
            if i % 4 == 3:
                output_dict4[key] = values


    with open("card_sets1.json", "w") as json_file:
        json.dump(output_dict1, json_file)

    with open("card_sets2.json", "w") as json_file:
        json.dump(output_dict2, json_file)

    with open("card_sets3.json", "w") as json_file:
        json.dump(output_dict3, json_file)

    with open("card_sets4.json", "w") as json_file:
        json.dump(output_dict4, json_file)

if __name__ == '__main__':
    main()