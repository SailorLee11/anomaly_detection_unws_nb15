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


keras_verbosity = 2
training_df = pd.read_csv("./data/NSLKDD/KDDTrain+.csv", header=None)
testing_df = pd.read_csv("./data/NSLKDD/KDDTest+.csv", header=None)

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
# 把数据集的头部放上去
training_df.columns = columns
testing_df.columns = columns

print("Training set has {} rows.".format(len(training_df)))
print("Testing set has {} rows.".format(len(testing_df)))

# 获取列表中的流量的类型
training_outcomes=training_df["outcome"].unique()
testing_outcomes=testing_df["outcome"].unique()

print("\nThe training set has {} possible outcomes \n".format(len(training_outcomes)) )
print(", ".join(training_outcomes)+".")
print("\nThe testing set has {} possible outcomes \n".format(len(testing_outcomes)))
print(", ".join(testing_outcomes)+".")

dos_attacks=["back","land","neptune","smurf","teardrop","pod","apache2","udpstorm","processtable","mailbomb"]
r2l_attacks=["snmpgetattack","snmpguess","worm","httptunnel","named","xlock","xsnoop","sendmail","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]
u2r_attacks=["sqlattack","buffer_overflow","loadmodule","perl","rootkit","xterm","ps","httptunnel"]
probe_attacks=["ipsweep","nmap","portsweep","satan","saint","mscan"]

classes=["Normal","Dos","R2L","U2R","Probe"]
def label_attack (row):
    if row["outcome"] in dos_attacks:
        return classes[1]
    if row["outcome"] in r2l_attacks:
        return classes[2]
    if row["outcome"] in u2r_attacks:
        return classes[3]
    if row["outcome"] in probe_attacks:
        return classes[4]
    return classes[0]

# 测试集的长度，这个没有用
test_samples_length = len(testing_df)
print(test_samples_length)

# 测试集和训练集进行合并
df=pd.concat([training_df,testing_df])
# axis = 1是按行来，调用label_attack
df["Class"]=df.apply(label_attack,axis=1) #当一个函数的参数存在于一个元组或者一个字典中时，用来间接的调用这个函数，并肩元组或者字典中的参数按照顺序传递给参数
#
df=df.drop("outcome",axis=1)
df=df.drop("difficulty",axis=1)

#将训练集和测试集重新分隔开
training_df= df.iloc[:-test_samples_length, :]
testing_df= df.iloc[-test_samples_length:,:]