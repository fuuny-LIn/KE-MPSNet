import pickle as pk
import pandas as pd
import torch


class MyDataSet:
    def __init__(self):
        signal_path = ''
        img_path = ''
        label_path = ''

        with open(signal_path, 'rb') as f:
            self.X1 = pk.load(f)
        f.close()

        with open(img_path, 'rb') as f:
            self.X2 = pk.load(f)
        f.close()

        self.Y = pd.read_csv(label_path, header=None).values

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        eeg = self.X1[idx, :, :]
        ecg = self.X2[idx, :]
        label = self.Y[idx, :].item()

        eeg = torch.tensor(eeg, dtype=torch.float32)
        ecg = torch.tensor(ecg, dtype=torch.float32)

        one_hot_label = torch.tensor(label, dtype=torch.long)
        return eeg, ecg, one_hot_label
