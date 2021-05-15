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
from util.plt_metrcis import plt_matrix,roc_auc,distribution,violinplot,boxplot,plot_auc
from util.get_data import make_test_data,load_data,load_malware_data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

def main():

    # set parameters
    # args = parse_args()
    # load and prepare data
    X_train = load_data('./data/unsw/Normal.csv',44)
    # X_fuzzer = load_data('./data/unsw/Fuzzers.csv', 44)
    x_malware = load_malware_data()
    x_test, y_test = make_test_data(X_train, 5000, x_malware)
    print("x_shape:",x_test.shape)
    print("y_shape:",y_test.shape)


    pr_scores = []
    roc_scores = []

    clf = svm.OneClassSVM( kernel='rbf',gamma='auto')
    X_train = X_train[1:1000, :]
    clf.fit(X_train)
    # x_test, y_test = make_test_data(X_train,2000,X_fuzzer)
    y_pred_train = clf.predict(x_test)
    scores = clf.decision_function(x_test).ravel() * (-1)


    # calc_metrics(X_test,scores)
    # roc_auc, prc_auc = calc_metrics(X_test, scores)

    print("预测的数据类别：",y_pred_train)
    print("scores",scores)


    plt_matrix(y_test,y_pred_train,'ocsvm')
    distribution(scores,'svm')
    # violinplot(scores,y_test,'svm')
    boxplot(scores,y_test,'svm')
    plot_auc(y_test,scores)


if __name__ == '__main__':
    main()