import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os


# 데이터셋 클래스 정의
class PascalVOCDataset(Dataset):
    def __init__(self, root, txt_file, image_transform=None, label_transform=None):
        self.root = root
        self.image_paths, self.label_paths = self.read_txt(txt_file)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def read_txt(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        image_paths = []
        label_paths = []
        for line in lines:
            image_name = line.strip()
            image_path = os.path.join(self.root, 'JPEGImages', f'{image_name}.jpg')
            label_path = os.path.join(self.root, 'SegmentationClass', f'{image_name}.png')
            image_paths.append(image_path)
            label_paths.append(label_path)
        return image_paths, label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert('L')
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        label = torch.squeeze(label).long()  # 레이블을 1D 텐서로 변환
        return image, label


def dataloader(batch_size):
    # 데이터 전처리
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    label_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 경로 설정
    root = 'VOC2012/'
    train_txt_path = 'VOC2012/pbl_train.txt'
    val_txt_path = 'VOC2012/pbl_val.txt'

    train_dataset = PascalVOCDataset(root, train_txt_path, image_transform=image_transform, label_transform=label_transform)
    val_dataset = PascalVOCDataset(root, val_txt_path, image_transform=image_transform, label_transform=label_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
