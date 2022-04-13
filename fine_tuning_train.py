import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import metrics
import torch.utils.data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(model, train_loader, loss_func, optimizer, epoch):

    epoch_loss = 0
    epoch_counter = 0
    # switch model to train mode (dropout enabled)
    model.train()


    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
            # send data to cuda
        if torch.cuda.is_available():
            data1, data2, target = data1.cuda(), data1.cuda(), target.cuda()
        optimizer.zero_grad()
        score1 = model(data1)
        score2 = model(data2)
        score3 = model(data3)
        loss = loss_func(score1, score2, score3)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item() * data1.shape[0]
        epoch_counter += float(data1.shape[0])

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data1), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


    return epoch_loss, epoch_counter

def test(model, test_loader):
    # switch model to eval model (dropout becomes pass through)

    model.eval()

    with torch.no_grad():
        for data1, data2, target in test_loader:
            # send data to cuda
            if torch.cuda.is_available():
                data1, data2, target = data1.cuda(),data1.cuda(), target.cuda()
            score1 = model(data1)
            score2 = model(data2)
            score = score2 - score1
            # distance = torch.sum(torch.abs(score),(1,2,3))
            distance = torch.sum(torch.abs(score), 1)

    threshold = 1000000
    fpr, tpr, _ = metrics.roc_curve(target, (threshold-distance)/threshold)
    roc_auc = metrics.auc(fpr, tpr)

    print('\nTest set: auc: {:.06f}\n'.format(
        roc_auc))
    return roc_auc

class TripletDataset():
    def __init__(self, dataset_pair, dataset_negative, transform=None):
        self.dataset_pair = dataset_pair
        self.dataset_negative = dataset_negative
        self.transform = transform

    def __len__(self):
        return len(self.dataset_pair)

    def __getitem__(self, idx):
        anchor, positive , target1 = self.dataset_pair[idx]
        negative, dataelse, target2 = self.dataset_negative[idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative



def main():
    # seed pytorch random number generator for reproducablity
    torch.manual_seed(2)

    train_dataset = torchvision.datasets.LFWPairs(
        './data', split='train',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    test_dataset = torchvision.datasets.LFWPairs(
        './data', split='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    # Split the indices in a stratified way
    indices1 = np.arange(0,1000)
    indices2 = np.arange(1100, 2100)

    # Warp into Subsets and DataLoaders
    train_dataset_pair = torch.utils.data.Subset(train_dataset, indices1)
    train_dataset_negative = torch.utils.data.Subset(train_dataset, indices2)

    train_dataset_triplet=TripletDataset(train_dataset_pair, train_dataset_negative, transform=None)

    # temp_img1, temp_img2, temp_img3 = train_dataset_triplet[100]
    # fig = plt.figure(figsize=(21, 7))
    # plt.subplot(1, 3, 1)
    # plt.imshow(temp_img1.numpy().transpose((1,2,0)))
    # plt.axis('off')
    # plt.title('Anchor')
    # plt.subplot(1, 3, 2)
    # plt.imshow(temp_img2.numpy().transpose((1,2,0)))
    # plt.axis('off')
    # plt.title('Positive')
    # plt.subplot(1, 3, 3)
    # plt.imshow(temp_img3.numpy().transpose((1,2,0)))
    # plt.axis('off')
    # plt.title('Negative')
    # plt.show()
    # fig.savefig('sample_dataset.png')

    train_loader = torch.utils.data.DataLoader(train_dataset_triplet, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # model = torchvision.models.alexnet(pretrained=True).features
    # model = torchvision.models.vgg16(pretrained=True).features
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    # send model parameters to cuda
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0006) #0.002 0.0006
    loss_func = nn.TripletMarginLoss(margin=10.0, p=1)

    epochs = 10
    train_loss = [0.1]*epochs
    test_accuracy = [0.1]*epochs

    for epoch in range(epochs):

        loss, epoch_counter = train(model, train_loader, loss_func, optimizer, epoch)

        train_loss[epoch]=loss / epoch_counter*1.0

        roc_auc = test(model, test_loader)

        test_accuracy[epoch]=roc_auc

    print('Saving model to Alex_w.pt')
    torch.save(model.state_dict(), 'Alex_w.pt')

    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1,2,1)
    plt.plot(train_loss)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Train Loss during Training', fontsize=16)
    plt.xticks(range(epochs))
    plt.grid('on')
    plt.legend(fontsize=14)

    plt.subplot(1,2,2)
    plt.plot(test_accuracy)
    plt.xticks(range(epochs))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    plt.title('Test AUC during Training', fontsize=16)
    plt.grid('on')

    fig.savefig('Alex_w.png')
    print('Saving image to Alex_w.png')

if __name__ == "__main__":
    main()
