import warnings
warnings.filterwarnings('ignore')
import os, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torchvision
import data_utils
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x, idx, noise):
        x = self.encoder(x)
        x = x.detach().numpy()
        x[:, idx] = x[:, idx] + noise
        x = torch.tensor(x).to(device)
        y = self.decoder(x)
        return x, y


def model_training(autoencoder, train_loader, epoch):
    loss_metric = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    autoencoder.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, _ = data
        images = Variable(images)
        images = images.view(images.size(0), -1)
        if cuda: images = images.to(device)
        outputs = autoencoder(images)
        loss = loss_metric(outputs, images)
        loss.backward()
        optimizer.step()
        if (i + 1) % LOG_INTERVAL == 0:
            print('Epoch [{}/{}] - Iter[{}/{}], MSE loss:{:.4f}'.format(
                epoch + 1, EPOCHS, i + 1, len(train_loader.dataset) // BATCH_SIZE, loss.item()
            ))


def evaluation(autoencoder, test_loaders):
    loss_metric = nn.MSELoss()
    autoencoder.eval()
    loss_avg_per_class = []
    z_acc = []
    for i, test_loader in enumerate(test_loaders): # for each class
        total_loss_per_class=0
        for data in test_loader:  # for each mini batch
            images, labels = data
            images = Variable(images)
            images = images.view(images.size(0), -1)
            if cuda: images = images.to(device)
            loss_noise = np.zeros((32, 1))
            for idx in range(32):
                z, outputs = autoencoder(images,idx=idx, noise=NOISE) # add noise to each dimension
                loss = loss_metric(outputs, images)
                loss_noise[idx] += loss.detach().numpy()
            # loss_per_batch = np.average(loss_noise)
            total_loss_per_class += loss_noise * len(images)
        loss_avg_per_class.append(total_loss_per_class /len(test_loader.dataset))
    avg_loss = sum(map(sum, loss_avg_per_class))/32/10.
    avg_loss = avg_loss.squeeze()
    fig, ax = plt.subplots()
    for i in range(10):
        plt.plot(loss_avg_per_class[i], label=i)
        ax.annotate(i, xy=(31, loss_avg_per_class[i][-1]),
                    xytext=(1.02 * 31, loss_avg_per_class[i][-1]))

    plt.xlabel("Dimension")
    plt.ylabel("Loss")
    # plt.title("Loss when weights are perturbed")
    plt.legend(loc='upper center', ncol=5)
    # plt.savefig('./images/noise_loss.png')

    print('\nAverage MSE Loss on Test set: {:.4f}'.format(avg_loss))

    global BEST_VAL
    if TRAIN_SCRATCH and avg_loss < BEST_VAL:
        BEST_VAL = avg_loss
        torch.save(autoencoder.state_dict(), './history/simple_autoencoder.pt')
        print('Save Best Model in HISTORY\n')


if __name__ == '__main__':

    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    LOG_INTERVAL = 100
    TRAIN_SCRATCH = False        # whether to train a model from scratch
    BEST_VAL = float('inf')     # record the best val loss
    NOISE = 0

    print('Noise: ', NOISE)

    train_loader, test_loader = data_utils.load_mnist(BATCH_SIZE)
    test_loader_class = data_utils.load_mnist_single_class(BATCH_SIZE)
    autoencoder = Autoencoder()
    if cuda: autoencoder.to(device)

    if TRAIN_SCRATCH:
        # Training autoencoder from scratch
        for epoch in range(EPOCHS):
            starttime = datetime.datetime.now()
            model_training(autoencoder, train_loader, epoch)
            endtime = datetime.datetime.now()
            print(f'Train a epoch in {(endtime - starttime).seconds} seconds')
            # evaluate on test set and save best model
            evaluation(autoencoder, test_loader_class)
        print('Training Complete with best validation loss {:.4f}'.format(BEST_VAL))

    else:
        autoencoder.load_state_dict(torch.load('./history/simple_autoencoder.pt'))
        # print(autoencoder.decoder[0].weight.shape)
        evaluation(autoencoder, test_loader_class)

        autoencoder.cpu()
        dataiter = iter(test_loader)
        images, _ = next(dataiter)
        images = Variable(images[:20])
        z, outputs = autoencoder(images.view(images.size(0), -1), idx=4, noise=NOISE)

        plt.figure()
        plt.title('Latent Representation')
        plt.imshow(z.detach().numpy())
        plt.savefig('./images/latent.png')

        # plot and save original and reconstruction images for comparisons
        plt.figure()
        plt.subplot(121)
        plt.title('Original MNIST Images')
        data_utils.imshow(torchvision.utils.make_grid(images))
        plt.subplot(122)
        plt.title('Autoencoder Reconstruction')
        data_utils.imshow(torchvision.utils.make_grid(
            outputs.view(images.size(0), 1, 28, 28).data
        ))
        plt.savefig('./images/autoencoder_noisy.png')
