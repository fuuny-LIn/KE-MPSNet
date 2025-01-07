import numpy as np
import wfdb
import pywt
from collections import defaultdict
import pandas as pd

PATH = 'F:\DataSet\MIT-BIH'
numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
             '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
             '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
             '231', '232', '233', '234']
R_peak_dict = ['N', 'L', 'R', 'a', 'V', 'F', 'J', 'A', 's', 'E', 'j', '/', 'Q', '|', 'B', '?', 'e', 'n', 'f', 'r']


# 加载数据集
def loadData():
    dataSet = []
    target = []
    pos = []
    for item in numberSet:
        record = wfdb.rdrecord(PATH + '\\' + item, channel_names=['MLII'])
        data = record.p_signal.flatten()
        # denoise
        rdata = denoise(data)

        annotation = wfdb.rdann(PATH + '\\' + item, 'atr')
        Rlocation = annotation.sample
        Rclass = annotation.symbol

        dataSet.append(rdata)
        target.append(Rclass)
        pos.append(Rlocation)

    dataSet = np.array(dataSet)
    return dataSet, target, pos


# 小波去噪
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 噪声置零
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')

    return rdata


# 创建AAMI与MIT-BIH标签映射字典
def creat_dict():
    N_dict = ['N', '·', 'L', 'R', 'e', '']
    S_dict = ['A', 'a', 'J', 'S']
    V_dict = ['V', 'E']
    F_dict = ['F']
    Q_dict = ['/', 'f', 'Q']

    description_dict = defaultdict(list)
    for item in N_dict:
        description_dict[item].append('N')
    for item in S_dict:
        description_dict[item].append('S')
    for item in V_dict:
        description_dict[item].append('V')
    for item in F_dict:
        description_dict[item].append('F')
    for item in Q_dict:
        description_dict[item].append('Q')

    return description_dict


# 根据先验标签进行R峰分割
def R_Peak_Segmentation(data, target, pos):
    des_dict = creat_dict()
    X_data, Y_data = [], []
    for i, item in enumerate(target):
        for j, item2 in enumerate(target[i]):
            if item2 in R_peak_dict:
                if (pos[i][j] >= 60) and (pos[i][j] <= 64810):
                    X_data.append(data[i, pos[i][j] - 60:pos[i][j] + 191])
                    if target[i][j] in des_dict:
                        Y_data.append(des_dict[target[i][j]])
                    else:
                        Y_data.append('N')
            else:
                continue
    return X_data, Y_data


# 数据增强、归一化、分割并保存
def Save2train_test():
    dataSet, tar, pos = loadData()
    # X为数据集，Y为标签
    X, Y = R_Peak_Segmentation(dataSet, tar, pos)
    # 白噪声数据增强
    X, Y = Data_Augmentation(X, Y)
    # 随机打乱
    np.random.seed(1234)
    rnd = np.random.randn(len(X), 1)

    df1 = pd.DataFrame(X)
    df1['rnd'] = rnd
    df1 = df1.sort_values(by='rnd', ascending=True)
    df1 = df1.drop(axis=1, columns='rnd')

    df2 = pd.DataFrame(Y)
    df2['rnd'] = rnd
    df2 = df2.sort_values(by=['rnd'], ascending=True)
    df2 = df2.drop(axis=1, columns='rnd')

    # 分割保存
    div = int(0.8 * len(X))
    train = df1[0:div]
    train.to_csv('./train.csv')

    test = df1[div:len(X)]
    test.to_csv('./test.csv')

    train_label = df2[0:div]
    test_label = df2[div:len(X)]
    train_label.to_csv('./train_label.csv')
    test_label.to_csv('./test_label.csv')


# 计算增强比例
def count_scale_factor(data1, data2, data3, data4):
    scale = []
    len1 = len(data1)
    len2 = len(data2)
    len3 = len(data3)
    len4 = len(data4)

    maxsize = max([len1, len2, len3, len4])

    scale.append((maxsize // len1) * 10)
    scale.append((maxsize // len2) * 10)
    scale.append((maxsize // len3) * 10)
    scale.append((maxsize // len4) * 10)

    return scale


# 按比例数据增强
def Data_Augmentation(dataset, labels):
    # N:S:V:F:Q = 8033:100:552:88:418  ==>  8544:5740:7051:6319:4598
    tempS, tempV, tempF, tempQ = [], [], [], []
    for idx, i in enumerate(labels):
        if i[0] == 'S':
            tempS.append(dataset[idx])
        elif i[0] == 'V':
            tempV.append(dataset[idx])
        elif i[0] == 'F':
            tempF.append(dataset[idx])
        elif i[0] == 'Q':
            tempQ.append(dataset[idx])
        else:
            continue

    scale_factor = count_scale_factor(tempS, tempV, tempF, tempQ)

    for i in range(scale_factor[0]):
        for j in tempS:
            noise = np.random.randn(1, 251)
            noise = 0.4 * noise
            idx = np.random.randint(0, len(tempS))
            new_signal = noise + tempS[idx]
            dataset.append(new_signal[0])
            labels.append('S')

    for i in range(scale_factor[1]):
        for j in tempV:
            noise = np.random.randn(1, 251)
            noise = 0.4 * noise
            idx = np.random.randint(0, len(tempV))
            new_signal = noise + tempV[idx]
            dataset.append(new_signal[0])
            labels.append('V')

    for i in range(scale_factor[2]):
        for j in tempF:
            noise = np.random.randn(1, 251)
            noise = 0.4 * noise
            idx = np.random.randint(0, len(tempF))
            new_signal = noise + tempF[idx]
            dataset.append(new_signal[0])
            labels.append('F')

    for i in range(scale_factor[3]):
        for j in tempQ:
            noise = np.random.randn(1, 251)
            noise = 0.4 * noise
            idx = np.random.randint(0, len(tempQ))
            new_signal = noise + tempQ[idx]
            dataset.append(new_signal[0])
            labels.append('Q')

    return dataset, labels


Save2train_test()
