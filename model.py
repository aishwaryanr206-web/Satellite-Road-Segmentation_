import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class ResNetUNet(nn.Module):
    """ResNet-34 encoder with U-Net-style skip connections."""
    def __init__(self, pretrained=True, out_channels=1):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        enc = resnet34(weights=weights)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.center = ConvBlock(512, 256)
        self.up4 = ConvBlock(256 + 256, 256)
        self.up3 = ConvBlock(256 + 128, 128)
        self.up2 = ConvBlock(128 + 64, 64)
        self.up1 = ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x0 = self.stem(x)            # 64, H/2
        x1 = self.layer1(self.pool(x0)) # 64, H/4
        x2 = self.layer2(x1)         # 128, H/8
        x3 = self.layer3(x2)         # 256, H/16
        x4 = self.layer4(x3)         # 512, H/32
        y = self.center(x4)
        y = F.interpolate(y, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up4(torch.cat([y, x3], dim=1))
        y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up3(torch.cat([y, x2], dim=1))
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, x1], dim=1))
        y = F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, x0], dim=1))
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        # Resize original image to x0 resolution for a final high-resolution fusion.
        # Use a lightweight projected input feature to retain local texture.
        return self.head(y)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.float().flatten(1)
        tp = (probs * targets).sum(1)
        fp = (probs * (1 - targets)).sum(1)
        fn = ((1 - probs) * targets).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()
