"""
@Time    : 2021/5/13 21:08
-------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------
@FileName: get_data.py
@Software: PyCharm
"""
import numpy as np
import pandas as pd

def load_data(data_to_path,length):
    """load data
    data should be compressed in npz
    """
    data = pd.read_csv(data_to_path)

    data = (data.astype(np.float32) - 127.5) / 127.5
    data = data.values.reshape(data.shape[0], length)  # 变成矩阵格式
    # data = np.array(data).reshape(1, -1)
    data = np.delete(data, -1, axis=1) #去掉label标签

    return data
def make_test_data(x_normal, normal_num, x_malware):
    """make test data which has specified mixed rate(rate_anomaly_test).
    shuffle and concatenate normal and abnormal data"""
    x_test_normal = x_normal[0:normal_num, :]
    y_test_normal = np.ones((normal_num, 1), dtype=np.int)
    # 异常的离群样本点
    print("y_test_normal:",y_test_normal.shape)
    y_test_malware = np.ones((x_malware.shape[0],1),dtype=np.int)*(-1)
    print("y_test_malware:",y_test_malware.shape)

    # concatenate test normal data and test anomaly data
    x_test = np.vstack((x_test_normal,x_malware))
    y_test = np.vstack((y_test_normal,y_test_malware))

    return x_test, y_test