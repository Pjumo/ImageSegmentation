from models import u2net, unet, deeplabv3
import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


celoss = nn.CrossEntropyLoss()


def muti_celoss_func(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = celoss(d0, labels_v)
    loss1 = celoss(d1, labels_v)
    loss2 = celoss(d2, labels_v)
    loss3 = celoss(d3, labels_v)
    loss4 = celoss(d4, labels_v)
    loss5 = celoss(d5, labels_v)
    loss6 = celoss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss


class ConfigLoader:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

        if self.model_name == 'u2net':
            self.model = u2net.u2net_caller(num_classes)
        elif self.model_name == 'unet':
            self.model = unet.unet_caller(num_classes)
        elif self.model_name == 'deeplabv3plus_resnet50':
            self.model = deeplabv3.deeplabv3plus_resnet50(num_classes=num_classes)
        elif self.model_name == 'deeplabv3plus_resnet101':
            self.model = deeplabv3.deeplabv3plus_resnet101(num_classes=num_classes)
        else:
            self.model = None

        if self.model_name == 'u2net':
            self.loss_cnt = 7
        else:
            self.loss_cnt = 1

    def load_model(self):
        return self.model

    def load_optim(self):
        if self.model_name == 'u2net':
            return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.model_name == 'unet':
            return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.5, 0.99))
        elif self.model_name in ['deeplabv3plus_resnet50', 'deeplabv3plus_resnet101']:
            return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            return None

    def load_loss_func(self):
        if self.model_name == 'u2net':
            return muti_celoss_func
        elif self.model_name == 'unet':
            return celoss
        elif self.model_name in ['deeplabv3plus_resnet50', 'deeplabv3plus_resnet101']:
            return FocalLoss()  # Use FocalLoss for deeplabv3plus models
        else:
            return None
