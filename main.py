import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model_loader import load_model
from data_loader import dataloader
import wandb

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 20
batch_size = 8

config = {
    'epochs': num_epochs,
    'classes': 21,
    'batch_size': batch_size,
    'learning_rate': 1e-3,
    'dataset': 'VOC2012',
    'architecture': 'U2NET'
}


class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred = torch.argmax(F.softmax(pred, dim=0), dim=0).cpu().detach().flatten().numpy()
            label = label.flatten().cpu().detach().numpy()

            mask = (label >= 0) & (label < self.num_classes)
            category = np.bincount(
                label[mask] * self.num_classes + pred[mask],
                minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += category

    def get_result(self):
        conf_mat = self.confusion_matrix
        pa = np.diag(conf_mat).sum() / (conf_mat.sum() + 1e-7)
        iou = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat) + 1e-7)
        miou = np.nanmean(iou)

        return pa, miou


# 모델 학습 및 평가 코드
def train_model():
    wandb.init(project='PBL4', entity='Pjumo', config=config)
    model = load_model('u2net', num_classes=21).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    train_loader, val_loader = dataloader(batch_size)
    cnt_progress = len(train_loader) // 30
    wandb.watch(model, criterion, log='all', log_freq=10)

    for epoch in range(num_epochs):
        model.train()

        for cnt, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if cnt % cnt_progress == 0:
                print(f'\rEpoch {epoch + 1} [', end='')
                for prog in range(cnt // cnt_progress):
                    print('■', end='')
                for prog in range((len(train_loader) - cnt) // cnt_progress):
                    print('□', end='')
                print(']', end='')

        model.eval()
        metric = SegMetrics(num_classes=21)
        metric.reset()
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                metric.update(outputs, labels)

            val_loss /= len(val_loader)
            pa, miou = metric.get_result()

            wandb.log({'val_loss': val_loss, 'PA': pa, 'mIoU': miou}, step=epoch)
            print('====== Val loss : {:.4f} \tPA : {:.4f} \tmIoU : {:.4f} ======'.format(val_loss, pa, miou))


if __name__ == '__main__':
    # 모델 학습
    train_model()
