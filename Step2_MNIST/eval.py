# -*- coding:utf-8 -*-

from train import Classifier, load_data, Config
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

def main():
    config = Config()

    model = Classifier(config.in_feat, config.num_class).to(config.device)
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

        X[:, 0] = (X[:, 0]+1)/2
    
        fig, ax = plt.subplots(4, 5)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax = ax.flatten()

        for i in range(20):
            ax[i].axis('off')
            ax[i].set_title(f'pred {pred_y[i]}')
            ax[i].imshow(X[i][0].to('cpu'), cmap = 'gray')
        
        plt.show()
        
if __name__ == '__main__':
    main()