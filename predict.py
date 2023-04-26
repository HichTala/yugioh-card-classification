import torch
from PIL import Image
from torch import load
from torch.cuda import is_available
from torchvision import datasets
from torchvision import transforms as T

import queries.queries as queries
from src.resnet import ResNet
from src.tools import art_cropper
from src.transformations import final_data_transforms
from test import parse_command_line, prediction


def compute_label_list(folder_dataset):
    label_list = []
    for k in range(len(folder_dataset)):
        label_list.append(folder_dataset.imgs[k][0].split('/')[-2])
    return label_list


def image_preprocess(image_path, device):
    img = Image.open(image_path)
    img = art_cropper(img)

    img = T.ToTensor()(img).unsqueeze(0)

    queries = [final_data_transforms(img) for _ in range(5)]

    return torch.cat(queries, dim=0).to(device)


def predict(model, prototypes, n_queries, top_k, label_list, test_label, image_path, device):
    model.train()
    inputs = image_preprocess(image_path, device)

    output = model(inputs)

    pred = prediction(output, prototypes, n_queries, top_k)
    print(pred)

    print("Prediction for : \n{} is\n{}".format(test_label, label_list[pred]))
    print()


def main(args):
    if args.device is None:
        device = 'cuda' if is_available() else 'cpu'
    else:
        device = args.device

    folder_dataset = datasets.ImageFolder(root=args.data_path)
    label_list = compute_label_list(folder_dataset=folder_dataset)

    model = ResNet().to(device)
    print(args.model_path)

    state_dict = load(args.model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    prototypes = state_dict['prototypes']

    if args.image_path is None:
        image_path_list = queries.image_path_list
        labels = queries.labels
    else:
        image_path_list = [args.image_path]
        labels = [args.labels]

    for index, image_path in enumerate(image_path_list):
        predict(
            model=model,
            prototypes=prototypes,
            n_queries=args.n_queries,
            top_k=args.top_k,
            label_list=label_list,
            test_label=labels[index],
            image_path=image_path,
            device=device
        )


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
