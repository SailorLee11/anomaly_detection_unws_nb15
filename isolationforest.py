"""
@Time    : 2021/5/15 21:23
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: isolationforest.py
@Software: PyCharm
"""
from sklearn.ensemble import IsolationForest
from util.get_data import _normalization_process_data,load_malware_data,make_test_data
from util.plt_metrcis import plt_matrix,roc_auc,distribution,violinplot,boxplot,plot_auc
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn import metrics

def main():

    x_normal = _normalization_process_data(44,'./data/unsw/Normal.csv')

    IF = IsolationForest(n_estimators=900, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False,
                         n_jobs=-1,
                         random_state=42, verbose=0, warm_start=False).fit(x_normal)

    x_malware = load_malware_data()
    x_test, y_test = make_test_data(x_normal, 5000, x_malware)

    scores = IF.score_samples(x_test)
    IF_predict = IF.predict(x_test)
    print(scores)

    plt_matrix(y_test,IF_predict,'IsolationForest')
    distribution(scores,'IsolationForest')
    # violinplot(scores,y_test,'svm')
    boxplot(scores,y_test,'IsolationForest')
    plot_auc(y_test,scores,'IsolationForest')

    LABELS = ['malware','normal']
    print(classification_report(y_test, IF_predict, target_names= LABELS, digits=4))

if __name__ == '__main__':
    main()