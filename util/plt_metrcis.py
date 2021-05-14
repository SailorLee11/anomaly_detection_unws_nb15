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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pylab import rcParams

def plt_matrix(y_test,y_pred_train):
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
    plt.savefig('./conclusion/confusion_matrix.png')
    plt.show()