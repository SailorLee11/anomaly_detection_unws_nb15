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


def _normalization_process_data(length,path):
    """
    从csv中获取数据，进行归一化处理，再转成矩阵格式
    :param length:
    :param path:
    :return:矩阵
    """
    data = pd.read_csv(path)
    data = (data.astype(np.float32) - 127.5) / 127.5
    data = data.values.reshape(data.shape[0], length)
    data = np.delete(data, -1, axis=1) #去掉label标签
    return data

def _random(x, y):
    X_Y = np.concatenate((x, y), axis=1)
    np.random.shuffle(X_Y)

    # 切分
    return X_Y


def load_malware_data():
    """
    concatenate malware data
    :return:
    """
    data1 = _normalization_process_data(44,'./data/unsw/Analysis_data.csv')
    data2 = _normalization_process_data(44,'./data/unsw/Backdoor_data.csv')
    data3 = _normalization_process_data(44,'./data/unsw/DoS_data.csv')
    data4 = _normalization_process_data(44,'./data/unsw/Exploits.csv')
    data5 = _normalization_process_data(44,'./data/unsw/Fuzzers.csv')
    data6 = _normalization_process_data(44,'./data/unsw/Generic_data.csv')
    data7 = _normalization_process_data(44,'./data/unsw/Reconnaissance.csv')
    data8 = _normalization_process_data(44,'./data/unsw/Worms.csv')
    data9 = _normalization_process_data(44, './data/unsw/Shellcode.csv')
    data = np.vstack((data1,data2,data3,data4,data5,data6,data7,data8,data9))
    # data = np.random.shuffle(data)
    return data

def make_test_data(x_normal, normal_num, x_malware):
    """make test data which has specified mixed rate(rate_anomaly_test).
    shuffle and concatenate normal and abnormal data"""
    # x_normal = np.random.shuffle(x_normal)
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


# if __name__ == '__main__':
#     data = _normalization_process_data(44,'../data/unsw/Analysis_data.csv')
#     data = np.random.shuffle(data)