import numpy as np
import pandas as pd
from datetime import datetime
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import DataSet
from tqdm.auto import tqdm
import model
import prepocess

seed_num = 1234
torch.manual_seed(seed_num)
random.seed(seed_num)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_num)
today = datetime.now().strftime('%m-%d')
save_path = './result' + today + '/best_model.ckpt'

SiameseNetwork = model.SiameseNetwork()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SiameseNetwork.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=24, T_mult=2, eta_min=0.01)

batch_size = 256
n_epochs = 500
best_acc = 0

train_set = DataSet.MyDataSet()
data_lens = len(train_set)
train_lens = int(0.9 * data_lens)
val_lens = data_lens - train_lens
train_set, val_set = random_split(train_set, [train_lens, val_lens])

train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False
)

total_train_loss = []
total_train_acc = []
total_val_loss = []
total_val_acc = []

for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_acc = []

    for batch in tqdm(train_loader):
        signals, imgs, labels = batch

        logits = model(imgs.to(device), signals.to(device))

        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        acc = (logits.argmax(dim=1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_acc.append(acc)

    train_loss_per_epoch = sum(train_loss) / len(train_loss)
    train_acc_per_epoch = sum(train_acc) / len(train_acc)

    total_train_loss.append(train_loss_per_epoch)
    total_train_acc.append(train_acc_per_epoch)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss_per_epoch:.5f}, acc = {train_acc_per_epoch:.5f}")

    model.eval()
    val_loss = []
    val_acc = []

    for batch in tqdm(val_loader):
        signals, imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device), signals.to(device))

        loss = criterion(logits, labels.to(device))

        acc = (logits.argmax(dim=1) == labels.to(device)).float().mean()

        val_loss.append(loss.item())
        val_acc.append(acc)

    valid_loss_per_epoch = sum(val_loss) / len(val_loss)
    valid_acc_per_epoch = sum(val_acc) / len(val_acc)

    total_val_loss.append(valid_loss_per_epoch)
    total_val_acc.append(valid_acc_per_epoch)
    print(f"[ Validation | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    if val_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), save_path)
        best_acc = val_acc

    prepocess.loss_figure(types='loss', data_1=total_train_loss, data_2=total_val_loss, n_epochs=n_epochs, save_path=save_path)
    prepocess.loss_figure(types='acc', data_1=total_train_acc, data_2=total_val_acc, n_epochs=n_epochs, save_path=save_path)
