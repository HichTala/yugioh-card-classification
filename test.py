import argparse

from torch import load, cat
from torch.cuda import is_available
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.prototypical_loss import euclidean_dist
from src.resnet import ResNet
from train import data_initialization


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Card classification testing parser', add_help=True)

    parser.add_argument('model_path', type=str,
                        help="Path to trained model to test")

    parser.add_argument('--data_path', default='./cardDatabaseFull/', type=str,
                        help="Path to training dataset's directory")

    parser.add_argument('--image_path', type=str, default=None,
                        help="Path to the image on which to make the inference "
                             "(default: Inference on all images listed in queries.py)")
    parser.add_argument('--labels', type=str, default=None,
                        help="Label of the given inference image (default: None)")

    parser.add_argument('--device', type=str, default=None,
                        help="device to use for training (default: cuda if available cpu otherwise)")

    parser.add_argument('--n_queries', type=int, default=5,
                        help="Number of query examples per classes (default: 5)")
    parser.add_argument('--n_classes', type=int, default=4864,
                        help="Number of classes in the dataset (default: 4864)")

    parser.add_argument('--top_k', type=int, default=10,
                        help="k best prediction to take into account (default:10)")


    return parser.parse_args()


def prediction(output, prototypes, n_queries, k):
    dist = euclidean_dist(output, prototypes)
    log_p_y = F.log_softmax(-dist, dim=1)

    top_k = {}

    for i in range(n_queries):
        for j, index in enumerate((log_p_y[i]).topk(k)[1].cpu().numpy()):
            if index not in top_k:
                top_k.update({index: 10 - j})
            else:
                top_k[index] += 10 - j

    return max(top_k, key=top_k.get)


def test(
        model,
        test_dataloader,
        prototypes,
        n_queries,
        n_classes,
        top_k,
        device
):
    results = 0
    bad_recognition = []

    print("Start testing")
    model.train()
    for batch in tqdm(test_dataloader, desc="\033[1mTest\033[0m", colour='green'):
        labels = batch['label']
        queries = batch['queries'].to(device)

        inputs = cat([
            queries.view(n_queries, *queries.size()[2:])
        ], dim=0)

        outputs = model(inputs)

        result = labels.item() == prediction(output=outputs, prototypes=prototypes, n_queries=n_queries, k=top_k)
        results += result

        if result == 0:
            bad_recognition.append(labels.item())

    print("\033[1m{0}\033[0m correctly recognized cards - \033[1m\033[96maccuracy\033[0m: {1:.2f}%".format(
        results,
        (results / n_classes) * 100
    ))
    print("Bad recognition:")
    print(bad_recognition)


def main(args):
    if args.device is None:
        device = 'cuda' if is_available() else 'cpu'
    else:
        device = args.device

    test_dataset = data_initialization(
        directory=args.data_path,
        n_classes=args.n_classes,
        n_supports=0,
        n_queries=args.n_queries,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)

    model = ResNet().to(device)

    state_dict = load(args.model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    prototypes = state_dict['prototypes'].to(device)

    test(
        model=model,
        test_dataloader=test_dataloader,
        prototypes=prototypes,
        n_queries=args.n_queries,
        n_classes=args.n_classes,
        top_k=args.top_k,
        device=device
    )


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
