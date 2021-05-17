"""
@Time    : 2021/5/17 15:30
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: AE.py
@Software: PyCharm
"""
import torch
import visdom
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
from util.get_data import _normalization_process_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

class AE(nn.Module):
    """
    自编码部分
    """
    def __init__(self,input_size):
        super(AE, self).__init__()
        self.input_size  = input_size
        self.encoder = nn.Sequential(
            # [b, 784] => [b, 256]
            nn.Linear(input_size, 32),
            nn.ReLU(),
            # [b, 256] => [b, 64]
            nn.Linear(32, 8),
            nn.ReLU(),
            # [b, 64] => [b, 20]
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            nn.Linear(4, 8),
            nn.ReLU(),
            # [b, 64] => [b, 256]
            nn.Linear(8, 32),
            nn.ReLU(),
            # [b, 256] => [b, 784]
            nn.Linear(32, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, -1)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape

        return x

def calculate_losses(x, preds):
    losses = np.zeros(len(x))
    for i in range(len(x)):
        losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)

    return losses


def main():
    """
    制作数据集.,X_train用作训练数据集的正常流量；X_nom_test用做测试数据集的正常流量
    :return:
    """
    X_nom = _normalization_process_data(44,'./data/unsw/Normal.csv')
    X_train, X_nom_test = train_test_split(X_nom, train_size=0.85, random_state=1)
    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_nom_test)
    train_set = TensorDataset(X_train)
    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)

    input_size = X_train.shape[1]

    model = AE(input_size)
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    loss_func = nn.MSELoss()
    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0.
        for step, (x,) in enumerate(train_loader):
            x_recon = model(x)
            # loss = calculate_losses(x_recon,x)
            loss = loss_func(x_recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
        total_loss /= len(train_set)

        print('Epoch {}/{} : loss: {:.4f}'.format(
            epoch + 1, num_epochs, loss))

    # mnist_train = datasets.MNIST('mnist', train=True, transform=transforms.Compose([
    #     transforms.ToTensor()
    # ]), download=True)
    # mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)
    #
    # mnist_test = datasets.MNIST('mnist', train=False, transform=transforms.Compose([
    #     transforms.ToTensor()
    # ]), download=True)
    # mnist_test = DataLoader(mnist_test, batch_size=32)
    #
    # epochs = 1000
    # lr = 1e-3
    # model = AE()
    # criteon = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # print(model)
    #
    # viz = visdom.Visdom()
    # for epoch in range(epochs):
    #     # 不需要label，所以用一个占位符"_"代替
    #     for batchidx, (x, _) in enumerate(mnist_train):
    #         x_hat = model(x)
    #         loss = criteon(x_hat, x)
    #
    #         # backprop
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     if epoch % 10 == 0:
    #         print(epoch, 'loss:', loss.item())
    #     x, _ = iter(mnist_test).next()
    #     with torch.no_grad():
    #         x_hat = model(x)
    #     viz.images(x, nrow=8, win='x', opts=dict(title='x'))
    #     viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()