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
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
from util.get_data import _normalization_process_data,load_malware_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from util.preprocess_data_NSL_KDD import
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)

class AE(nn.Module):
    """
    自编码部分
    """
    def __init__(self,input_size):
        super(AE, self).__init__()
        self.input_size  = input_size
        self.encoder = nn.Sequential(
            # [b, 784] => [b, 256]
            nn.Linear(input_size, 8),
            nn.Dropout(0.5),  # drop 50% neurons
            nn.ReLU(),
            # # [b, 256] => [b, 64]
            # nn.Linear(32, 8),
            # nn.Dropout(0.5),  # drop 50% neurons
            # nn.ReLU(),
            # # [b, 64] => [b, 20]
            # nn.Linear(8, 4),
            # nn.Dropout(0.5),  # drop 50% neurons
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            # nn.Linear(4, 8),
            # nn.Dropout(0.5),  # drop 50% neurons
            # nn.ReLU(),
            # # [b, 64] => [b, 256]
            # nn.Linear(8, 32),
            # nn.Dropout(0.5),  # drop 50% neurons
            # nn.ReLU(),
            # # [b, 256] => [b, 784]
            nn.Linear(8, input_size),
            nn.Dropout(0.5),  # drop 50% neurons
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        # flatten
        # -1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列，那不确定的地方就可以写成-1
        x = x.view(batchsz, -1)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape

        return x


# def get_model():



def calculate_losses(x, preds):
    losses = np.zeros(len(x))
    for i in range(len(x)):
        losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)

    return losses

def get_recon_err(X,model):
    return torch.mean((model(X)-X)**2,dim = 1).detach().numpy()

def main():
    """
    制作数据集.,X_train用作训练数据集的正常流量；X_nom_test用做测试数据集的正常流量
    :return:
    """
    # X_nom = _normalization_process_data(44,'./data/unsw/Normal.csv')
    # x_test_malware = load_malware_data()
    # X_train, X_nom_test = train_test_split(X_nom, train_size=0.85, random_state=1)
    # x_test = np.concatenate([x_test_malware,X_nom_test],axis = 0)
    # y_test = np.concatenate([np.ones(len(x_test_malware)), np.zeros(len(X_nom_test))])    # 制作label

    # 变准化处理
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(x_test)
    # # 转为tensor张量
    # X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    print("X_train",X_train.shape)
    print("X_test",X_test.shape)
    train_set = TensorDataset(X_train) #对tensor进行打包
    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)  #数据集放入Data.DataLoader中，可以生成一个迭代器，从而我们可以方便的进行批处理


    input_size = X_train.shape[1]

    model = AE(input_size)
    optimizer = torch.optim.Adam(model.parameters(), 0.00001)
    print(model)
    loss_func = nn.MSELoss(reduction='mean')
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0.
        for step, (x,) in enumerate(train_loader):
            x_recon = model(x)
            # loss = calculate_losses(x_recon,x)
            loss = loss_func(x_recon, x)
            optimizer.zero_grad()
            # 计算中间的叶子节点
            loss.backward()
            # 内容信息反馈
            optimizer.step()

            total_loss += loss.item() * len(x)
        total_loss /= len(train_set)

        print('Epoch {}/{} : loss: {:.4f}'.format(
            epoch + 1, num_epochs, loss))


    # 如何判断异常值？？？

    # threshold = model.history["loss"][-1]
    # testing_set_predictions = model.forward(x_test)
    # test_losses = calculate_losses(x_test, testing_set_predictions)
    # testing_set_predictions = np.zeros(len(test_losses))
    # testing_set_predictions[np.where(test_losses > threshold)] = 1



if __name__ == '__main__':
    main()