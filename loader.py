from models import u2net, unet, deeplabv3
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights


def cross_entropy_loss(logits, targets):
    loss = F.cross_entropy(logits, targets)
    return loss


def dice_coefficient(logits, targets, smooth=1):
    probs = torch.sigmoid(logits)
    num = (probs * targets).sum()
    denom = probs.sum() + targets.sum() + smooth
    dice = (2 * num + smooth) / denom
    return dice


def dice_loss(logits, targets, smooth=1):
    loss = 1 - dice_coefficient(logits, targets, smooth)
    return loss


def combined_loss(logits, targets, alpha=0.75):
    ce_loss = cross_entropy_loss(logits, targets)
    d_loss = dice_loss(logits, targets)
    loss = alpha * ce_loss + (1 - alpha) * d_loss
    return loss


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


class SobelHeuristicBlock(nn.Module):
    def __init__(self, in_channels):
        super(SobelHeuristicBlock, self).__init__()
        self.in_channels = in_channels

        # Define horizontal and vertical Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        # Add a channel dimension to the Sobel filters
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

        # Define convolutional layers with Sobel filters as weights
        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # Initialize the convolutional layers with Sobel filters
        self.conv_x.weight = nn.Parameter(self.sobel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(self.sobel_y, requires_grad=False)

    def forward(self, x):
        # Apply horizontal and vertical Sobel filters
        sobel_x = self.conv_x(x)
        sobel_y = self.conv_y(x)

        # Combine the outputs to form the final SHK output
        shk_output = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)

        return shk_output


class DeepLabV3WithSHB(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3WithSHB, self).__init__()
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=num_classes)
        self.shb = SobelHeuristicBlock(in_channels=num_classes)

    def forward(self, x):
        # Get the output from the DeepLabV3 model
        x = self.deeplabv3(x)['out']

        # Upsample the output to match the input size
        x = nn.functional.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        # Apply the SHB to the output
        x = self.shb(x)

        return {'out': x}


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
        elif self.model_name == 'deeplabv3plus_resnet101_pretrained':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, num_classes=num_classes)
        elif self.model_name == 'deeplabv3plus_resnet101_pretrained_sobel':
            self.model = DeepLabV3WithSHB(num_classes=num_classes)
        else:
            self.model = None

    def load_model(self):
        return self.model

    def load_optim(self):
        if self.model_name == 'u2net':
            return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.model_name == 'unet':
            return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.5, 0.99))
        elif self.model_name in ['deeplabv3plus_resnet50', 'deeplabv3plus_resnet101']:
            return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.model_name in ['deeplabv3plus_resnet101_pretrained', 'deeplabv3plus_resnet101_pretrained_sobel']:
            return torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            return None

    def load_loss_func(self):
        if self.model_name == 'u2net':
            return muti_celoss_func
        elif self.model_name == 'unet':
            return celoss
        elif self.model_name in ['deeplabv3plus_resnet50', 'deeplabv3plus_resnet101']:
            return FocalLoss()  # Use FocalLoss for deeplabv3plus models
        elif self.model_name in ['deeplabv3plus_resnet101_pretrained', 'deeplabv3plus_resnet101_pretrained_sobel']:
            return combined_loss
        else:
            return None
