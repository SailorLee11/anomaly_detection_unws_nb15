"""
@Time    : 2021/5/19 15:27
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: preprocess_data_NSL_KDD.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


# keras_verbosity = 2
# training_df = pd.read_csv("./data/NSLKDD/KDDTrain+.csv", header=None)
# testing_df = pd.read_csv("./data/NSLKDD/KDDTest+.csv", header=None)
#
columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'difficulty'
]
dos_attacks = ["back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm", "processtable", "mailbomb"]
r2l_attacks = ["snmpgetattack", "snmpguess", "worm", "httptunnel", "named", "xlock", "xsnoop", "sendmail", "ftp_write",
               "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]
u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps", "httptunnel"]
probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]
classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
# # 识别为5个大类
# classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
def _label_attack (row):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]

def _minmax_scale_values(training_df,testing_df, col_name):
    """
    对数值进行标准化处理
    :param training_df:
    :param testing_df:
    :param col_name:
    :return:
    """
    scaler = MinMaxScaler()
    # scaler = scaler.fit(training_df[col_name].reshape(-1, 1))
    train_values_standardized = scaler.fit_transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized

def _encode_text(training_df, testing_df, name):
    """
    用于将字符串转为 数值类型
    :param training_df:
    :param testing_df:
    :param name:
    :return:
    """
    # get_dummies 是用作one hot编码  参考https://blog.csdn.net/maymay_/article/details/80198468
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns:
            testing_df[dummy_name] = testing_set_dummies[x]
        else:
            testing_df[dummy_name] = np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)

def _process_sympolic_data(training_df, testing_df,df):
    """
    将数据集合中的 字符串变成 0 1 编码格式
    :param training_df:
    :param testing_df:
    :param df:
    :return:
    """
    sympolic_columns = ["protocol_type", "service", "flag"]
    label_column = "Class"
    for column in df.columns:
        if column in sympolic_columns:
            # 如果非数值格式的,采用 one_hot编码格式进行计算
            _encode_text(training_df, testing_df, column)
        elif not column == label_column:
            _minmax_scale_values(training_df, testing_df, column)
    return training_df,testing_df
# 把数据集的头部放上去

def disassemble_the_data_set(training_df, testing_df,columns):
    """
    把数据集的头部放上去,换标签
    先将数据集和测试集合并，然后将标签，几个小类合并为几个大类
    :param training_df:
    :param testing_df:
    :return:
    """

    training_df.columns = columns
    testing_df.columns = columns

    # columns = columns
    # dos_attacks = dos_attacks
    # r2l_attacks = r2l_attacks
    # u2r_attacks = u2r_attacks
    # probe_attacks = probe_attacks

    print("Training set has {} rows.".format(len(training_df)))
    print("Testing set has {} rows.".format(len(testing_df)))

    # 获取列表中的流量的类型
    training_outcomes=training_df["outcome"].unique()
    testing_outcomes=testing_df["outcome"].unique()

    print("\nThe training set has {} possible outcomes \n".format(len(training_outcomes)) )
    print(", ".join(training_outcomes)+".")
    print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
    print(", ".join(testing_outcomes)+".")

    # 测试集的长度
    test_samples_length = len(testing_df)
    print(test_samples_length)

    # 测试集和训练集进行合并
    df=pd.concat([training_df,testing_df])
    # axis = 1是按行来，调用label_attack  新增加Class标签
    df["Class"]=df.apply(_label_attack,axis=1) #当一个函数的参数存在于一个元组或者一个字典中时，用来间接的调用这个函数，并肩元组或者字典中的参数按照顺序传递给参数
    #
    df=df.drop("outcome",axis=1)
    df=df.drop("difficulty",axis=1)

    #将训练集和测试集重新分隔开
    training_df= df.iloc[:-test_samples_length, :]
    testing_df= df.iloc[-test_samples_length:,:]
    return df,training_df,testing_df

# print("The training set has {} possible outcomes \n".format(len(training_outcomes)) )
# print(", ".join(training_outcomes)+".")
# print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
# print(", ".join(testing_outcomes)+".")

# ------------------------------处理 非数值类型--------------------------------------


# training_df,testing_df = _process_sympolic_data(training_df, testing_df)


# sympolic_columns = ["protocol_type", "service", "flag"]
# label_column = "Class"
# for column in df.columns:
#     if column in sympolic_columns:
#         # 如果非数值格式的,采用 one_hot编码格式进行计算
#         _encode_text(training_df, testing_df, column)
#     elif not column == label_column:
#         _minmax_scale_values(training_df, testing_df, column)

# training_df.head(5)
# testing_df.head(5)

def _trainsfrom_tensor(training_df,testing_df):
    """
    转为tensor格式，最后进行输出用于训练
    :param training_df:
    :param testing_df:
    :return:
    """
    # y label    这里先处理训练集的文件 数据集
    x,y_train=training_df,training_df.pop("Class").values
    # dataframe 转为  矩阵格式
    x=x.values
    print(x)
    # 处理测试集的文件
    x_test,y_test=testing_df,testing_df.pop("Class").values
    print(x_test)
    x_test=x_test.values
    # 这里转为 tensor格式
    X_train, X_test = torch.FloatTensor(x), torch.FloatTensor(x_test)
    return X_train,X_test,y_train,y_test

# X_train,X_test,y_train,y_test = _trainsfrom_tensor(training_df,testing_df)

def make_combined_y_data(y_train,y_test,classes):
    """

    :param y_train:
    :param y_test:
    :return:
    """
    y_train=np.ones(len(y_train),np.int8)
    y_train[np.where(y_train==classes[0])]=0
    y_test=np.ones(len(y_test),np.int8)
    y_test[np.where(y_test==classes[0])]=0
    return y_train,y_test

def early_configuration():
    training_df = pd.read_csv("./data/NSLKDD/KDDTrain+.csv", header=None)
    testing_df = pd.read_csv("./data/NSLKDD/KDDTest+.csv", header=None)
    # columns = [
    #     #     'duration',
    #     #     'protocol_type',
    #     #     'service',
    #     #     'flag',
    #     #     'src_bytes',
    #     #     'dst_bytes',
    #     #     'land',
    #     #     'wrong_fragment',
    #     #     'urgent',
    #     #     'hot',
    #     #     'num_failed_logins',
    #     #     'logged_in',
    #     #     'num_compromised',
    #     #     'root_shell',
    #     #     'su_attempted',
    #     #     'num_root',
    #     #     'num_file_creations',
    #     #     'num_shells',
    #     #     'num_access_files',
    #     #     'num_outbound_cmds',
    #     #     'is_host_login',
    #     #     'is_guest_login',
    #     #     'count',
    #     #     'srv_count',
    #     #     'serror_rate',
    #     #     'srv_serror_rate',
    #     #     'rerror_rate',
    #     #     'srv_rerror_rate',
    #     #     'same_srv_rate',
    #     #     'diff_srv_rate',
    #     #     'srv_diff_host_rate',
    #     #     'dst_host_count',
    #     #     'dst_host_srv_count',
    #     #     'dst_host_same_srv_rate',
    #     #     'dst_host_diff_srv_rate',
    #     #     'dst_host_same_src_port_rate',
    #     #     'dst_host_srv_diff_host_rate',
    #     #     'dst_host_serror_rate',
    #     #     'dst_host_srv_serror_rate',
    #     #     'dst_host_rerror_rate',
    #     #     'dst_host_srv_rerror_rate',
    #     #     'outcome',
    #     #     'difficulty'
    #     # ]
    #     # dos_attacks = ["back", "land", "neptune", "smurf", "teardrop", "pod", "apache2", "udpstorm", "processtable",
    #     #                "mailbomb"]
    #     # r2l_attacks = ["snmpgetattack", "snmpguess", "worm", "httptunnel", "named", "xlock", "xsnoop", "sendmail",
    #     #                "ftp_write",
    #     #                "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]
    #     # u2r_attacks = ["sqlattack", "buffer_overflow", "loadmodule", "perl", "rootkit", "xterm", "ps", "httptunnel"]
    #     # probe_attacks = ["ipsweep", "nmap", "portsweep", "satan", "saint", "mscan"]
    #     # # 识别为5个大类
    #     # classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
    df,training_df,testing_df = disassemble_the_data_set(training_df, testing_df,columns)
    training_df,testing_df = _process_sympolic_data(training_df, testing_df,df)
    X_train,X_test,y_train,y_test = _trainsfrom_tensor(training_df, testing_df)
    y_train,y_test = make_combined_y_data(y_train, y_test, classes)
    return X_train,X_test,y_train,y_test

def main():
    X_train,X_test,y_train,y_test = early_configuration()

if __name__ == '__main__':
    main()