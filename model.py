import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from pyts.image import GramianAngularField


one_hot = ['J', 'AFL', 'AFIB', 'N']


def Seq2Img(data):
    graf = GramianAngularField(image_size=128, method='s')
    result = np.ndarray((1, 1, 128, 128))
    timestamp = np.arange(0, len(data))
    signals = np.vstack((data, timestamp))
    img = graf.transform(signals)[0]
    result[0, 0, :, :] = img
    return result


def Data_Normalization(data: np.ndarray):
    max_value = np.max(data)
    min_value = np.min(data)
    Dvalue = max_value - min_value
    for idx, i in enumerate(data):
        data[idx] = (i - min_value) / Dvalue
    if np.max(data) > 1 or np.min(data) < -1:
        raise Exception("Error Normalization")
    return data


def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=6)
    cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def signal_preporcess(data):
    data = denoise(data[320:1600])
    data = Data_Normalization(data)
    return data


def read_data(signal, device):
    x1 = signal_preporcess(signal)
    x2 = Seq2Img(x1)
    x1 = x1.reshape((1, 1, -1))
    x1 = torch.tensor(x1, dtype=torch.float32, device=device)
    x2 = torch.tensor(x2, dtype=torch.float32, device=device)
    return x1, x2


def cal_res(res):
    idx = res.argmax(dim=1)
    l = one_hot[idx]
    if l == 'N':
        return '该信号平稳，无异常情况。'
    else:
        return '该信号可能存在心律不齐现象！'


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=[1, 1], padding=1) -> None:
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.layer(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, Res_Block) -> None:
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(Res_Block, 64, [[1, 1], [1, 1]])
        self.layer2 = self.make_layer(Res_Block, 128, [[1, 1], [1, 1]])
        self.layer3 = self.make_layer(Res_Block, 256, [[1, 1], [1, 1]])

    def make_layer(self, block, channels, strides):
        layers = []
        for s in strides:
            layers.append(block(self.in_channel, channels, s))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 2)
        return out


class ResBlock_Wang(nn.Module):
    def __init__(self, in_c, out_c, stride=1, padding='same') -> None:
        super(ResBlock_Wang, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=8, stride=stride, padding=padding),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=5, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_c),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        out = self.layer(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet_Wang(nn.Module):
    def __init__(self, Res_Block_Wang) -> None:
        super(ResNet_Wang, self).__init__()
        self.in_channel = 1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
        )
        self.layer1 = self.make_layer(Res_Block_Wang, 64, [1, 1, 1], 'same')
        self.layer2 = self.make_layer(Res_Block_Wang, 128, [1, 1, 1], 'same')
        self.layer3 = self.make_layer(Res_Block_Wang, 128, [1, 1, 1], 'same')

    def make_layer(self, block, channels, strides, padding):
        layers = []
        for s in strides:
            layers.append(block(self.in_channel, channels, s, padding))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out_l1 = self.layer1(out)
        out_l2 = self.layer2(out_l1)
        out_l3 = self.layer3(out_l2)
        result = F.adaptive_avg_pool1d(out_l3, 32)
        return result


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # input size: [Batch, 1, 128, 128]
        self.CNN_Branch = ResNet(ResBlock)
        # input size: [Batch, 1, 1920]
        self.Temporal_Branch = ResNet_Wang(ResBlock_Wang)

        self.sp_tp_attention_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 128, 1, bias=False)
        )
        self.sp_tp_attention_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(128, 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 128, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.Converge = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Flatten(),
            nn.Linear(5408, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b1_0 = self.CNN_Branch(y)
        b2_0 = self.Temporal_Branch(x)
        b2_0 = torch.unsqueeze(b2_0, 3)
        avg_a_0 = self.sp_tp_attention_avg(b1_0)
        max_a_0 = self.sp_tp_attention_max(b1_0)
        a_v_0 = self.sigmoid(b2_0 + avg_a_0 + max_a_0)
        branch_0 = b1_0.mul(a_v_0)

        out = self.Converge(branch_0)
        return out


