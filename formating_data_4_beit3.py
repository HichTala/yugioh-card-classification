import os

from datasets import ImageNetDataset

data_path = os.path.join("./data", "club_yugioh_dataset")
card_types = ['spell', 'trap', 'effect', 'normal', 'fusion', 'ritual']

for card_type in card_types:
    ImageNetDataset.make_dataset_index(
        train_data_path = os.path.join(data_path, card_type, "train"),
        val_data_path = os.path.join(data_path, card_type, "val"),
        index_path = os.path.join(data_path, card_type),
    )
