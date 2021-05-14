"""
@Time    : 2021/5/13 11:09
--------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
--------------------------------
@FileName: ocsvm.py
@Software: PyCharm
"""

import argparse
from collections import namedtuple
import sys
from util.plt_metrcis import plt_matrix,roc_auc,distribution,violinplot,boxplot
from util.get_data import make_test_data,load_data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

# def load_data(data_to_path,length):
#     """load data
#     data should be compressed in npz
#     """
#     data = pd.read_csv(data_to_path)
#
#     data = (data.astype(np.float32) - 127.5) / 127.5
#     data = data.values.reshape(data.shape[0], length)  # 变成矩阵格式
#     # data = np.array(data).reshape(1, -1)
#     data = np.delete(data, -1, axis=1) #去掉label标签
#
#     return data

def prepare_data(full_images, full_labels, normal_label, rate_normal_train, TRIN_RAND_SEED):
    """prepare data
    split data into anomaly data and normal data
    """
    TRAIN_DATA_RNG = np.random.RandomState(TRIN_RAND_SEED)

    # data whose label corresponds to anomaly label, otherwise treated as normal data
    ano_x = full_images[full_labels != normal_label]
    ano_y = full_labels[full_labels != normal_label]
    normal_x = full_images[full_labels == normal_label]
    normal_y = full_labels[full_labels == normal_label]

    # replace label : anomaly -> 1 : normal -> 0
    ano_y[:] = 1
    normal_y[:] = 0

    # shuffle normal data and label
    inds = TRAIN_DATA_RNG.permutation(normal_x.shape[0])
    normal_x_data = normal_x[inds]
    normal_y_data = normal_y[inds]

    # split normal data into train and test
    index = int(normal_x.shape[0] * rate_normal_train)
    trainx = normal_x_data[:index]
    testx_n = normal_x_data[index:]
    testy_n = normal_y_data[index:]

    split_data = namedtuple('split_data', ('train_x', 'testx_n', 'testy_n', 'ano_x', 'ano_y'))

    return split_data(
        train_x=trainx,
        testx_n=testx_n,
        testy_n=testy_n,
        ano_x=ano_x,
        ano_y=ano_y
    )

# def make_test_data(x_normal, normal_num, x_malware):
#     """make test data which has specified mixed rate(rate_anomaly_test).
#     shuffle and concatenate normal and abnormal data"""
#     x_test_normal = x_normal[0:normal_num, :]
#     y_test_normal = np.ones((normal_num, 1), dtype=np.int)
#     # 异常的离群样本点
#     print("y_test_normal:",y_test_normal.shape)
#     y_test_malware = np.ones((x_malware.shape[0],1),dtype=np.int)*(-1)
#     print("y_test_malware:",y_test_malware.shape)
#
#     # concatenate test normal data and test anomaly data
#     x_test = np.vstack((x_test_normal,x_malware))
#     y_test = np.vstack((y_test_normal,y_test_malware))
#     # ano_x = split_data.ano_x
#     # ano_y = split_data.ano_y
#     # testx_n = split_data.testx_n
#     # testy_n = split_data.testy_n
#
#     # anomaly data in test
#     # inds_1 = RNG.permutation(ano_x.shape[0])
#     # ano_x = ano_x[inds_1]
#     # ano_y = ano_y[inds_1]
#
#     # index_1 = int(testx_n.shape[0] * rate_anomaly_test)
#     # testx_a = ano_x[:index_1]
#     # testy_a = ano_y[:index_1]
#
#
#     # testx = np.concatenate([testx_a, testx_n], axis=0)
#     # testy = np.concatenate([testy_a, testy_n], axis=0)
#
#     return x_test, y_test

def calc_metrics(testy, scores):
    precision, recall, _ = precision_recall_curve(testy, scores)
    roc_auc = roc_auc_score(testy, scores)
    prc_auc = auc(recall, precision)

    return roc_auc, prc_auc


def main():

    # set parameters
    # args = parse_args()
    # load and prepare data
    X_train = load_data('./data/unsw/Normal.csv',44)
    X_fuzzer = load_data('./data/unsw/Fuzzers.csv', 44)
    x_test, y_test = make_test_data(X_train, 2000, X_fuzzer)
    print("x_shape:",x_test.shape)
    print("y_shape:",y_test.shape)


    pr_scores = []
    roc_scores = []

    clf = svm.OneClassSVM( kernel='rbf',gamma='auto')
    X_train = X_train[1:8000, :]
    clf.fit(X_train)
    # x_test, y_test = make_test_data(X_train,2000,X_fuzzer)
    y_pred_train = clf.predict(x_test)
    scores = clf.decision_function(x_test).ravel() * (-1)


    # calc_metrics(X_test,scores)
    # roc_auc, prc_auc = calc_metrics(X_test, scores)

    print("预测的数据类别：",y_pred_train)
    print("scores",scores)
    # print("roc_auc:",roc_auc,"   prc_auc:",prc_auc)
    # y_test = np.ones((x_test.shape[0], 1), dtype=np.int) * (-1)

    plt_matrix(y_test,y_pred_train,'ocsvm')
    distribution(scores,'svm')
    # violinplot(scores,y_test,'svm')
    boxplot(scores,y_test,'svm')
    # from sklearn.metrics import confusion_matrix
    # conf_matrix = confusion_matrix(y_test, y_pred_train)
    # conf_matrix_new = conf_matrix
    # print(conf_matrix_new)
    # import seaborn as sns
    # from pylab import rcParams
    # sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    # RANDOM_SEED = 42
    # LABELS = ["-1", "1"]
    #
    # plt.figure(figsize=(20, 15))
    # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    # plt.title("Traffic Classification Confusion Matrix")
    # plt.ylabel('Application traffic samples')
    # plt.xlabel('Application traffic samples')
    # plt.savefig('confusion_matrix.png')
    # plt.show()
    # --------------------------------------------------
    # nu : the upper limit ratio of anomaly data(0<=nu<=1)
    # nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # train model and evaluate with changing parameter nu
    # for nu in nus:
    #     # train with nu
    #     clf = svm.OneClassSVM(nu=nu, kernel='rbf',  gamma='auto')
    #     clf.fit(X_train)
    #     y_pred_train = clf.predict(X_test)
    #     scores = clf.decision_function(X_test).ravel() * (-1)
    #     # print(y_pred_train)
    #     # clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
    #     # clf.fit(split_data.train_x)
    #     calc_metrics()
    #     total_pr = 0
    #     total_roc = 0
    #
    #     # repeat test by randomly selected data and evaluate
    #
    #     print('--- nu : ', nu, ' ---')
    #     print('PR AUC : ', total_pr)
    #     print('ROC_AUC : ', total_roc)
    #
    # print('***' * 5)
    # print('PR_AUC MAX : ', max(pr_scores))
    # print('ROC_AUC MAX : ', max(roc_scores))
    # print('ROC_MAX_NU : ', nus[int(np.argmax(roc_scores))])


if __name__ == '__main__':
    main()