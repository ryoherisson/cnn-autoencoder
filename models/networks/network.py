import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super(SimpleCNN, self).__init__()

        channels = 16

        self.cbnr1 = Conv2DBatchNormRelu(in_channels, channels, 3, 1, 0)
        self.cbnr2 = Conv2DBatchNormRelu(channels, channels*2, 3, 1, 0)
        self.cbnr3 = Conv2DBatchNormRelu(channels*2, channels*4, 3, 1, 0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = torch.nn.Dropout2d(p=0.3)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(in_features=channels*4*13*13, out_features=512)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x):
        x = self.cbnr1(x)
        x = self.cbnr2(x)
        x = self.cbnr3(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


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

