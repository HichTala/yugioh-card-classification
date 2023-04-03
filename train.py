from torch import optim, save, load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet101_Weights

from sm_dataset import SiameseNetworkDataset
from siamese_network import SiameseNetwork
from loss_functions import ContrastiveLoss
from transformations import data_transforms


class Config:
    training_dir = "./cardDatabaseFull/"

    train_batch_size = 7
    train_number_epochs = 300
    resum_training = True


if __name__ == '__main__':

    folder_dataset = datasets.ImageFolder(root=Config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=data_transforms)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  batch_size=Config.train_batch_size)

    # net = SiameseNetwork(weights=None).cuda()
    # net.load_state_dict(load("models/res-withShift-no_para-150-072020.pth"))

    net = SiameseNetwork(ResNet101_Weights.IMAGENET1K_V1).cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0
    epoch = 0

    if Config.resum_training:
        state_dict = load("./models/checkpoints/res_epoch_60_012023.pth")
        epoch = state_dict['epoch']
        net.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        loss_history = state_dict['loss']

    while epoch <= Config.train_number_epochs:
        for i, data in enumerate(train_dataloader):

            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(input0, input1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch number {} - Iteration number {}\n Current loss {}\n".format(epoch, i,
                                                                                         loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
            # Adding a validation set

        savePath = './models/checkpoints/res_epoch_{}_012023.pth'.format(epoch)
        save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history
        }, savePath)

        epoch += 1

    savePath = './res-300-normalized.pth'
    save(net.state_dict(), savePath)
