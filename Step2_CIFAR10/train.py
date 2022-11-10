# -*- coding:utf-8 -*-

import pathlib
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt

DEBUG = False

# PRETRAINED = True

import sys, getopt

'''
    create modules used for classifer

    get path:
        import pathlib
        print(pathlib.Path(__file__).parent.absolute())
'''

def load_data(path):
    '''
        imputs:
            path : str for store the dataset
        return:
            mnist_train:
            mnist_test:
    '''
    
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    
    mnist_train = CIFAR10(root=path, train=True, download=False, transform=train_tf)
    mnist_test = CIFAR10(root=path, train=False, download=False, transform=test_tf)

    return mnist_train, mnist_test 


class Config():
    batch_size = 128
    in_feat = 28*28
    num_class = 10
    lr = 1e-3
    n_epoch = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    path = pathlib.Path(__file__).parent.absolute()
    data_path = str(path) + '/data/'
    model_path = str(path) + '/model/model.pt'

    pretrained = True

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block4= nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.block1(x) + x)

        x = self.conv2(x)
        x = F.relu(self.block2(x) + x)

        x = self.conv3(x)
        x = F.relu(self.block3(x) + x)

        x = self.conv4(x)
        x = F.relu(self.block4(x) + x)

        y = F.adaptive_avg_pool2d(self.pool(x), (1, 1))
        y = y.squeeze()

        return self.fc(y)


def debug():
    config = Config()
    train, test = load_data(config.data_path)
    
    train_loader = DataLoader(train, config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, config.batch_size, shuffle=False, num_workers=2)

    X, y = next(iter(train_loader))
    tf = transforms.ToPILImage()
    plt.imshow(tf(X[0]))
    plt.show()


def main():

    config = Config()
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"h",["lr=","epoches=", "help"])
    except getopt.GetoptError:
        print('[usage]')
        print('\ttrain.py\n\t(optional) --lr <learning rate(defult 1e-3)>\n\t(optional) --epoches <num of epoch(defult 100)>\n\t(optional) --outpath <path model to save model(defult  Step2/model)>\n\t(optional) -p <use pretrained model>')
        print('\ttrain.py -h (--help) get helps')
        return
    
    for opt, arg in (opts):
        if opt == '-h' or opt == '--help':
            print('[usage]')
            print('\ttrain.py\n\t(optional) --lr <learning rate(defult 1e-3)>\n\t(optional) --epoches <num of epoch(defult 100)>\n\t(optional) --outpath <path model to save model(defult  Step2/model)>\n\t(optional) -p <use pretrained model>')
            return
        elif opt == '--lr':
            config.lr = int(arg)
        elif opt == '--epoches':
            config.n_epoch = int(arg)
        elif opt == '--outpath':
            config.model_path = arg
        elif opt == '-p':
            config.pretrained = True


    
    mnist_train, mnist_test = load_data(config.data_path)
    
    train_loader = DataLoader(mnist_train, config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(mnist_test, config.batch_size, shuffle=False, num_workers=2)

    # X, y = next(iter(train_loader))
    # print(X[0][0])
    # print(y)

    # plt.imshow(X[0][0], cmap = 'gray')
    # plt.show()

    model = Classifier().to(config.device)
    if config.pretrained:
        model.load_state_dict(torch.load(config.model_path))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader), gamma=0.99)

    for epoch in range(config.n_epoch):
        model.train()
        bar = tqdm(train_loader, total=len(train_loader))
        for X, y in bar:
            X = X.to(config.device)
            y = y.to(config.device)

            pred = model(X)

            _, pred_y = pred.max(dim = 1)
            acc = (pred_y == y).sum().item()/len(y)

            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            bar.set_description_str(f'Epoch:{epoch+1:>03d}')
            bar.set_postfix_str(f'acc:{acc:>.4f} loss:{loss.item():>.4f} lr{lr_scheduler.get_last_lr()[0]:>.5f}')

        model.eval()
        with torch.no_grad():
            acc = 0
            for X, y in test_loader:
                X = X.to(config.device)    
                y = y.to(config.device)

                pred = model(X)
                _, pred_y = pred.max(dim = 1)

                acc += (pred_y == y).sum().item()

            acc /= len(mnist_test)
            print(f'test acc : {acc:>.4f}')
        torch.save(model.state_dict(), config.model_path)


if __name__ == "__main__" and not DEBUG:
    main()

elif DEBUG:
    debug()