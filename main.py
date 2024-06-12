import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from loader import ConfigLoader
from data_loader import dataloader
import wandb

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 20
batch_size = 16

config = {
    'epochs': num_epochs,
    'classes': 21,
    'batch_size': batch_size,
    'dataset': 'VOC2012',
    'architecture': 'deeplabv3+'
}

model_name = 'deeplabv3plus_resnet101_pretrained'
loader = ConfigLoader(model_name=model_name, num_classes=21)
wandb_name = 'deeplabv3plus_resnet101_pretrained_0.00001_16_CRF'


def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        ignore_index: int
):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=labels.device)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret


# Fully Connected CRFs
def dense_crf(image, output_probs):
    h, w = image.shape[:2]
    n_labels = output_probs.shape[0]

    # Get unary potentials
    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_softmax(output_probs)
    d.setUnaryEnergy(unary)

    # Create pairwise Gaussian potentials
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Create pairwise bilateral potentials
    image = np.ascontiguousarray(image, dtype=np.uint8)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(5)
    map = np.argmax(Q, axis=0).reshape((h, w))

    return map


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


# 이미지 출력 코
def visualize_results(image, output_probs, crf_result):
    output_before_crf = np.argmax(output_probs, axis=0)

    image = np.clip(image, 0, 255)

    # 시각화
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Dense CRF 적용 전
    ax[1].imshow(output_before_crf)
    ax[1].set_title('Before Dense CRF')
    ax[1].axis('off')

    # Dense CRF 적용 후
    ax[2].imshow(crf_result)
    ax[2].set_title('After Dense CRF')
    ax[2].axis('off')

    plt.show()


# 모델 학습 및 평가 코드
def train_model():
    wandb.init(project='PBL4', entity='Pjumo', name=wandb_name, config=config)
    model = loader.load_model().to(device)
    optimizer = loader.load_optim()
    criterion = loader.load_loss_func()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_loader, val_loader = dataloader(batch_size)
    cnt_progress = len(train_loader) // 30
    wandb.watch(model, criterion, log='all', log_freq=10)

    if loader.model_name.find('pretrained') != -1:
        for param in model.parameters():
            param.requires_grad = True

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for cnt, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.int64)
            labels = label_to_one_hot_label(labels, 21, ignore_index=255)
            criterion = criterion
            optimizer.zero_grad()

            if loader.model_name == 'u2net':
                outputs, d1, d2, d3, d4, d5, d6 = model(images)
                loss = criterion(outputs, d1, d2, d3, d4, d5, d6, labels)
            elif loader.model_name.find('pretrained') != -1:
                outputs = model(images)['out']
                loss = criterion(outputs, labels)
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

                if loader.model_name == 'u2net':
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

                elif loader.model_name.find('pretrained') != -1:
                    outputs = model(images)['out']
                    if val_loss == 0:
                        output_ = outputs.detach().cpu().numpy()[7]
                        output_concat = np.argmax(output_, axis=0)
                        img = Image.fromarray(np.uint8(output_concat), mode='P')

                        palette_with_alpha_values = []
                        for i in range(768):
                            color = palette[i].cpu().numpy()[7]
                            palette_with_alpha_values.append(color)

                        img.putpalette(palette_with_alpha_values, "RGB")
                    loss = criterion(outputs, labels_one_hot)

                    # Dense CRF 적용
                    for i in range(images.shape[0]):
                        image = images[i].cpu().numpy().transpose(1, 2, 0)
                        output_probs = F.softmax(outputs[i], dim=0).cpu().numpy()

                        image = (image * 255).astype(np.uint8)
                        crf_result = dense_crf(image, output_probs)
                        crf_result = torch.tensor(crf_result, dtype=torch.int64, device=device)
                        labels[i] = crf_result

                else:
                    outputs = model(images)
                    if val_loss == 0:
                        output_ = outputs.detach().numpy()[7]
                        output_concat = np.argmax(output_, axis=0)
                        img = Image.fromarray(np.uint8(output_concat), mode='P')
                        palette_with_alpha_values = []

                        for i in range(768):
                            color = palette[i].numpy()[7]
                            palette_with_alpha_values.append(color)
                        img.putpalette(palette_with_alpha_values, "RGB")
                    loss = criterion(outputs, labels_one_hot)

                val_loss += loss.item()

                metric.update(outputs, labels)

            val_loss /= len(val_loader)
            pa, miou = metric.get_result()

            rand_idx = np.random.randint(0, len(images))
            image = images[rand_idx].cpu().numpy().transpose(1, 2, 0)
            output_probs = F.softmax(outputs[rand_idx], dim=0).cpu().numpy()

            image_scaled = (image * 255).astype(np.uint8)
            crf_result = dense_crf(image_scaled, output_probs)
            crf_result_tensor = torch.tensor(crf_result, dtype=torch.int64, device=device)

            wandb.log({'img': wandb.Image(img), 'val_loss': val_loss, 'PA': pa, 'mIoU': miou}, step=epoch)
            print('====== Val loss : {:.4f} \tPA : {:.4f} \tmIoU : {:.4f} ======'.format(val_loss, pa, miou))
            visualize_results(image, output_probs, crf_result)

        scheduler.step(total_train_loss)


if __name__ == '__main__':
    train_model()
