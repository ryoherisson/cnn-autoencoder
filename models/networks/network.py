import torch
import torch.nn as nn


class CNNAutoEncoder(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super(CNNAutoEncoder, self).__init__()

        channels = 16

        self.encoder = nn.Sequential(
            Conv2DBatchNormRelu(in_channels, channels, 3, 1, 0),
            Conv2DBatchNormRelu(channels, channels*2, 3, 1, 0),
            Conv2DBatchNormRelu(channels*2, channels*4, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            UpConv2DBatchNormRelu(channels*4, channels*2, 3, 1, 0),
            UpConv2DBatchNormRelu(channels*2, channels, 3, 1, 0),
            UpConv2DBatchNormRelu(channels, in_channels, 3, 1, 0),
            nn.Sigmoid(),
        )

        self.classifier_fc = nn.Sequential(
            nn.Linear(in_features=channels*4*13*13, out_features=512),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=64),
            nn.Linear(in_features=64, out_features=n_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        ae_out = self.decoder(x)

        x = x.view(x.size(0), -1)
        clf_out = self.classifier_fc(x)

        return clf_out, ae_out


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class UpConv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpConv2DBatchNormRelu, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x
