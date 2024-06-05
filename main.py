import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from loader import ConfigLoader
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
    'dataset': 'VOC2012',
    'architecture': 'UNET'
}

loader = ConfigLoader(model_name='deeplabv3plus_resnet50', num_classes=21)
wandb_name = 'deeplabv3plus_resnet50'


def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        ignore_index: int
):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret


class SegMetrics():
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
                label[mask] * self.num_classes + pred[mask]
                , minlength=self.num_classes ** 2
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
    wandb.init(project='PBL4', entity='Pjumo', name=wandb_name, config=config)
    model = loader.load_model().to(device)
    optimizer = loader.load_optim()
    criterion = loader.load_loss_func()

    train_loader, val_loader = dataloader(batch_size)
    cnt_progress = len(train_loader) // 30
    wandb.watch(model, criterion, log='all', log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for cnt, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.int64)
            labels = label_to_one_hot_label(labels, 21, ignore_index=255)
            criterion = criterion
            optimizer.zero_grad()

            if loader.loss_cnt == 7:
                outputs, d1, d2, d3, d4, d5, d6 = model(images)
                loss = criterion(outputs, d1, d2, d3, d4, d5, d6, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            if cnt % cnt_progress == 0:
                print(f'\rEpoch {epoch + 1} [', end='')
                for prog in range(cnt // cnt_progress):
                    print('■', end='')
                for prog in range((len(train_loader) - cnt) // cnt_progress):
                    print('□', end='')
                print(']', end='')

        total_train_loss /= len(train_loader)
        print(f' - Train loss: {total_train_loss:.4f}')

        model.eval()
        metric = SegMetrics(num_classes=21)
        metric.reset()
        with torch.no_grad():
            val_loss = 0
            for images, labels, palette in val_loader:
                images = images.to(device)
                labels = labels.to(device, dtype=torch.int64)
                labels_one_hot = label_to_one_hot_label(labels, 21, ignore_index=255)

                if loader.loss_cnt == 7:
                    outputs, d1, d2, d3, d4, d5, d6 = model(images)
                    if val_loss == 0:
                        output_ = outputs[0].numpy()
                        output_concat = np.argmax(output_, axis=1)
                        img = Image.fromarray(np.uint8(output_concat), mode='P')
                        palette_with_alpha_values = []
                        for i in range(768):
                            color = palette[i].numpy()[0]
                            palette_with_alpha_values.append(color)
                        img.putpalette(palette_with_alpha_values, "RGB")
                    loss = criterion(outputs, d1, d2, d3, d4, d5, d6, labels_one_hot)
                else:
                    outputs = model(images)
                    if val_loss == 0:
                        output_ = outputs.detach().numpy()[0]
                        output_concat = np.argmax(output_, axis=0)
                        img = Image.fromarray(np.uint8(output_concat), mode='P')
                        palette_with_alpha_values = []
                        for i in range(768):
                            color = palette[i].numpy()[0]
                            palette_with_alpha_values.append(color)
                        img.putpalette(palette_with_alpha_values, "RGB")
                    loss = criterion(outputs, labels_one_hot)

                val_loss += loss.item()

                metric.update(outputs, labels)

            val_loss /= len(val_loader)
            pa, miou = metric.get_result()

            wandb.log({'img': wandb.Image(img), 'val_loss': val_loss, 'PA': pa, 'mIoU': miou}, step=epoch)
            print('====== Val loss : {:.4f} \tPA : {:.4f} \tmIoU : {:.4f} ======'.format(val_loss, pa, miou))


if __name__ == '__main__':
    train_model()
