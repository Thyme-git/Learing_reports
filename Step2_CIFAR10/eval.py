# -*- coding:utf-8 -*-

from train import Classifier, load_data, Config
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

labels = ['airplane',
          'automobile',
          'bird',
          'cat',
          'deer',
          'dog',
          'frog',
          'horse',
          'ship',
          'truck']

def main():
    config = Config()

    model = Classifier().to(config.device)
    model.load_state_dict(torch.load(config.model_path))

    _, mnist_test = load_data(config.data_path)

    test_loader = DataLoader(mnist_test, config.batch_size, shuffle=True, num_workers=2)

    model.eval()
    it = iter(test_loader)
    with torch.no_grad():
        X, y = next(it)
        X = X.to(config.device)
        y = y.to(config.device)

        pred = model(X)
        pred_y = pred.argmax(dim = 1)

        mean = [0.507, 0.487, 0.441]
        std = [0.267, 0.256, 0.276]
        X[:, 0] = X[:, 0]*std[0] + mean[0]
        X[:, 1] = X[:, 1]*std[1] + mean[1]
        X[:, 2] = X[:, 2]*std[2] + mean[2]
    
        fig, ax = plt.subplots(4, 5)
        fig.set_size_inches((10, 10))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
        ax = ax.flatten()

        for i in range(20):
            ax[i].axis('off')
            ax[i].set_title(f'{labels[pred_y[i]]}')
            ax[i].imshow(X[i].to('cpu').transpose(0, 2).transpose(0, 1))
        
        plt.show()
        
if __name__ == '__main__':
    main()