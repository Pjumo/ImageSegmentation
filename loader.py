from models import u2net, unet
import torch.nn as nn
import torch

celoss = nn.CrossEntropyLoss()


def muti_celoss_func(d0, d1, d2, d3, d4, d5, d6, labels_v):
    d0 = torch.argmax(d0, dim=1).float()
    d1 = torch.argmax(d1, dim=1).float()
    d2 = torch.argmax(d2, dim=1).float()
    d3 = torch.argmax(d3, dim=1).float()
    d4 = torch.argmax(d4, dim=1).float()
    d5 = torch.argmax(d5, dim=1).float()
    d6 = torch.argmax(d6, dim=1).float()

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
            return torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.99))
        else:
            return None

    def load_loss_func(self):
        if self.model_name == 'u2net':
            return muti_celoss_func
        elif self.model_name == 'unet':
            return celoss
        else:
            return None
