"""
@Time    : 2021/5/12 22:02
--------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
--------------------------------
@FileName: preprocess_classes.py
@Software: PyCharm
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,normalize

train = pd.read_csv('data/unsw/UNSW_NB15_training-set.csv')
test  = pd.read_csv('./data/unsw/UNSW_NB15_testing-set.csv')
le1 = LabelEncoder()
le = LabelEncoder()

def process_class_malware():

    combined_data = pd.concat([train, test]).drop(['id'],axis=1)
    Generic_data = combined_data[combined_data['attack_cat'] == 'Generic']
    # Generic_data.to_csv('data/unsw/Generic_data_raw.csv',index=False)
    vector1 = Generic_data['attack_cat']
    Generic_data['attack_cat'] = le1.fit_transform(vector1)
    Generic_data['proto'] = le.fit_transform(Generic_data['proto'])
    Generic_data['service'] = le.fit_transform(Generic_data['service'])
    Generic_data['state'] = le.fit_transform(Generic_data['state'])
    Generic_data.to_csv('data/unsw/Generic_data.csv',index=False)

    Analysis_data = combined_data[combined_data['attack_cat'] == 'Analysis']
    vector2 = Analysis_data['attack_cat']
    Analysis_data['attack_cat'] = le1.fit_transform(vector2)
    Analysis_data['proto'] = le.fit_transform(Analysis_data['proto'])
    Analysis_data['service'] = le.fit_transform(Analysis_data['service'])
    Analysis_data['state'] = le.fit_transform(Analysis_data['state'])
    Analysis_data.to_csv('data/unsw/Analysis_data.csv',index=False)

    Backdoor_data = combined_data[combined_data['attack_cat'] == 'Backdoor']
    vector3 = Backdoor_data['attack_cat']
    Backdoor_data['attack_cat'] = le1.fit_transform(vector3)
    Backdoor_data['proto'] = le.fit_transform(Backdoor_data['proto'])
    Backdoor_data['service'] = le.fit_transform(Backdoor_data['service'])
    Backdoor_data['state'] = le.fit_transform(Backdoor_data['state'])
    Backdoor_data.to_csv('data/unsw/Backdoor_data.csv', index=False)

    DoS_data = combined_data[combined_data['attack_cat'] == 'DoS']
    vector4 = DoS_data['attack_cat']
    DoS_data['attack_cat'] = le1.fit_transform(vector4)
    DoS_data['proto'] = le.fit_transform(DoS_data['proto'])
    DoS_data['service'] = le.fit_transform(DoS_data['service'])
    DoS_data['state'] = le.fit_transform(DoS_data['state'])
    DoS_data.to_csv('data/unsw/DoS_data.csv', index=False)

    Exploits_data = combined_data[combined_data['attack_cat'] == 'Exploits']
    vector5 = Exploits_data['attack_cat']
    Exploits_data['attack_cat'] = le1.fit_transform(vector5)
    Exploits_data['proto'] = le.fit_transform(Exploits_data['proto'])
    Exploits_data['service'] = le.fit_transform(Exploits_data['service'])
    Exploits_data['state'] = le.fit_transform(Exploits_data['state'])
    Exploits_data.to_csv('data/unsw/Exploits.csv', index=False)

    Fuzzers = combined_data[combined_data['attack_cat'] == 'Fuzzers']
    vector6 = Fuzzers['attack_cat']
    Fuzzers['attack_cat'] = le1.fit_transform(vector6)
    Fuzzers['proto'] = le.fit_transform(Fuzzers['proto'])
    Fuzzers['service'] = le.fit_transform(Fuzzers['service'])
    Fuzzers['state'] = le.fit_transform(Fuzzers['state'])
    Fuzzers.to_csv('data/unsw/Fuzzers.csv', index=False)

    Reconnaissance = combined_data[combined_data['attack_cat'] == 'Reconnaissance']
    vector7 = Reconnaissance['attack_cat']
    Reconnaissance['attack_cat'] = le1.fit_transform(vector7)
    Reconnaissance['proto'] = le.fit_transform(Reconnaissance['proto'])
    Reconnaissance['service'] = le.fit_transform(Reconnaissance['service'])
    Reconnaissance['state'] = le.fit_transform(Reconnaissance['state'])
    Reconnaissance.to_csv('data/unsw/Reconnaissance.csv', index=False)

    Shellcode = combined_data[combined_data['attack_cat'] == 'Shellcode']
    vector8 = Shellcode['attack_cat']
    Shellcode['attack_cat'] = le1.fit_transform(vector8)
    Shellcode['proto'] = le.fit_transform(Shellcode['proto'])
    Shellcode['service'] = le.fit_transform(Shellcode['service'])
    Shellcode['state'] = le.fit_transform(Shellcode['state'])
    Shellcode.to_csv('data/unsw/Shellcode.csv', index=False)

    Worms = combined_data[combined_data['attack_cat'] == 'Worms']
    vector9 = Worms['attack_cat']
    Worms['attack_cat'] = le1.fit_transform(vector9)
    Worms['proto'] = le.fit_transform(Worms['proto'])
    Worms['service'] = le.fit_transform(Worms['service'])
    Worms['state'] = le.fit_transform(Worms['state'])
    Worms.to_csv('data/unsw/Worms.csv', index=False)

    Normal = combined_data[combined_data['attack_cat'] == 'Normal']
    vector10 = Normal['attack_cat']
    Normal['attack_cat'] = le1.fit_transform(vector10)
    Normal['proto'] = le.fit_transform(Normal['proto'])
    Normal['service'] = le.fit_transform(Normal['service'])
    Normal['state'] = le.fit_transform(Normal['state'])
    Normal.to_csv('data/unsw/Normal.csv', index=False)


if __name__ == '__main__':
    process_class_malware()
