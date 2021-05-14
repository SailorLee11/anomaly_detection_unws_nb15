"""
@Time    : 2021/5/13 20:55
--------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
--------------------------------
@FileName: plt_metrcis.py
@Software: PyCharm
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams

def plt_matrix(y_test,y_pred_train,description):
    conf_matrix = confusion_matrix(y_test, y_pred_train)
    conf_matrix_new = conf_matrix
    print(conf_matrix_new)
    import seaborn as sns
    from pylab import rcParams
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    RANDOM_SEED = 42
    LABELS = ["-1", "1"]

    plt.figure(figsize=(20, 15))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Traffic Classification Confusion Matrix")
    plt.ylabel('Application traffic samples')
    plt.xlabel('Application traffic samples')
    plt.savefig('./conclusion/confusion_matrix_%s.png'%(description))
    plt.show()

def roc_auc(y_test, scores, pos_label=1, show=False, path=None):
    """Compute ROC curve and ROC area for each class"""
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    if show:
        # Equal Error Rate
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        plt.figure()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        lw = 2
        plt.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=lw,
                 label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1 - eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # if path:
        #     mkdir(os.path.dirname(path))
        plt.savefig(path + "_roc_auc.png")
        plt.show()
    #         plt.close()
    return {'roc_auc': roc_auc}

#         distribution(score, self.getname())
def distribution(y, name):
    # RANDOM_SEED = 42
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    sns.kdeplot(y, shade=True, ax=ax)
    plt.savefig('./conclusion/distribution_%s.png' % (name))
    plt.show()


def scatterplot(x, y):
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    data = np.concatenate([x, y], axis=1)
    df = pd.DataFrame(data, columns=["x", "y"])
    sns.scatterplot(x="x", y="y", data=df)
    # sns.jointplot(x="x", y="y", data=df)
    plt.show()


def violinplot(x, y,name):
    """
    这里x是横坐标，种类，y是纵坐标 异常的分数
    :param x:
    :param y:
    :return:
    """
    plt.figure(figsize=(24, 20))
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    data = np.concatenate([y, x], axis=1)
    df = pd.DataFrame(data, columns=["class", "score"])
    sns.violinplot(x="class", y="score", data=df, split=True)
    plt.savefig('./conclusion/violinplot_%s.png' % (name))
    plt.show()


def boxplot(x, y, name):
    """
    这里x是横坐标，种类，y是纵坐标
    :param x:
    :param y:
    :return:
    """
    plt.figure(figsize=(24, 20))
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    data = np.concatenate([y, x], axis=1)
    df = pd.DataFrame(data, columns=["class", "score"])
    sns.boxplot(x="class", y="score", data=df)
    plt.savefig('./conclusion/boxplot_%s.png' % (name))
    plt.show()
