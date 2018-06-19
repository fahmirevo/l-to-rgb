import torch.nn as nn
from .commons import Fire


class SqueezeNet(nn.Module):

    def __init__(self, in_planes=3, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            Fire(32, 16, 32, 32),
            Fire(64, 16, 32, 32),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            Fire(64, 32, 64, 64),
            Fire(128, 32, 64, 64),
            Fire(128, 32, 128, 128),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, n_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
        )

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        X = X.mean(-1).mean(-1)
        return self.final_activation(X)
