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
from util.preprocess_data_NSL_KDD import early_configuration
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score

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
        losses[i] = ((torch.from_numpy(preds[i]) - x[i]) ** 2).mean(axis=None)

    return losses

def get_recon_err(X,model):
    return torch.mean((model.forward(X)-X)**2,dim = 1).detach().numpy()

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

    X_train,X_test,y_train,y_test = early_configuration()
    print("X_train",X_train.shape)
    print("X_test",X_test.shape)
    train_set = TensorDataset(X_train) #对tensor进行打包
    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)  #数据集放入Data.DataLoader中，可以生成一个迭代器，从而我们可以方便的进行批处理


    input_size = X_train.shape[1]

    model = AE(input_size)
    optimizer = torch.optim.Adam(model.parameters(), 0.00001)
    print(model)
    loss_func = nn.MSELoss(reduction='mean')
    num_epochs = 30
    tmp_total_loss = 0
    for epoch in range(num_epochs):
        total_loss = 0.
        for step, (x,) in enumerate(train_loader):
            x_recon = model.forward(x)
            # loss = calculate_losses(x_recon,x)
            loss = loss_func(x_recon, x)
            optimizer.zero_grad()
            # 计算中间的叶子节点，计算图
            loss.backward()
            # 内容信息反馈
            optimizer.step()

            total_loss += loss.item() * len(x)
        total_loss /= len(train_set)
        if epoch>(num_epochs-5):
            tmp_total_loss +=loss
        print('Epoch {}/{} : loss: {:.4f}'.format(
            epoch + 1, num_epochs, loss))
    threshold = tmp_total_loss/5
    threshold = threshold.double()
    print(threshold)
    # 如何判断异常值？？？
    # model = model.eval()
    # print(model)
    # # net = net.eval()  # not needed - no dropout
    # X = torch.Tensor(X_test)  # all input item as Tensors
    # Y = model(X)  # all outputs as Tensors
    # N = len(X)
    # max_se = 0.0;
    # max_ix = 0
    # for i in range(N):
    #     curr_se = torch.sum((X[i] - Y[i]) * (X[i] - Y[i]))
    #     if curr_se.item() > max_se:
    #         max_se = curr_se.item()
    #         max_ix = i
    # threshold = model.history["loss"][-1]
    # testing_set_predictions = model.forward(X_test)
    # print(testing_set_predictions)
    # testing_set_predictions = testing_set_predictions.detach().numpy()

    recon_err_train = get_recon_err(X_train,model)
    recon_err_test = get_recon_err(X_test,model)
    recon_err = np.concatenate([recon_err_train, recon_err_test])
    labels = np.concatenate([np.zeros(len(recon_err_train)), y_test])
    index = np.arange(0, len(labels))

    sns.kdeplot(recon_err[labels == 0], shade=True)
    sns.kdeplot(recon_err[labels == 1], shade=True)
    plt.show()

    from sklearn.metrics import accuracy_score, f1_score

    threshold = np.linspace(0, 10, 500)
    acc_list = []
    f1_list = []

    for t in threshold:
        y_pred = (recon_err_test > t).astype(np.int)
        acc_list.append(accuracy_score(y_pred, y_test))
        f1_list.append(f1_score(y_pred, y_test))

    plt.figure(figsize=(8, 6))
    plt.plot(threshold, acc_list, c='y', label='acc')
    plt.plot(threshold, f1_list, c='b', label='f1')
    plt.xlabel('threshold')
    plt.ylabel('classification score')
    plt.legend()
    plt.show()

    i = np.argmax(f1_list)
    t = threshold[i]
    score = f1_list[i]
    print('Recommended threshold: %.3f, related f1 score: %.3f' % (t, score))

    y_pred = (recon_err_test > t).astype(np.int)
    FN = ((y_test == 1) & (y_pred == 0)).sum()
    FP = ((y_test == 0) & (y_pred == 1)).sum()
    print('In %d data of test set, FN: %d, FP: %d' % (len(y_test), FN, FP))

    # test_losses = get_recon_err(X_test, model)
    # print(type(test_losses))
    # testing_set_predictions = np.zeros(len(test_losses))
    # testing_set_predictions[np.where(test_losses > threshold)] = 1
    #
    # accuracy = accuracy_score(y_test, testing_set_predictions)
    # recall = recall_score(y_test, testing_set_predictions)
    # precision = precision_score(y_test, testing_set_predictions)
    # f1 = f1_score(y_test, testing_set_predictions)
    #
    # print("accuracy:",accuracy)
    # print("recall:",recall)
    # print("precision:",precision)
    # print("f1:",f1)

if __name__ == '__main__':
    main()