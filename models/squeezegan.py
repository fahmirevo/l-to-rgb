import torch.nn as nn
from .commons import Fire, DeFire, ResBlock


class Encoder(nn.Module):

    def __init__(self, in_planes=1):
        super().__init__()
        self.gateway = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        )

        self.fire_module1 = nn.Sequential(
            Fire(32, 16, 32, 32),
            Fire(64, 16, 32, 32),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        )

        self.fire_module2 = nn.Sequential(
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        )

    def forward(self, X):
        X = self.gateway(X)
        X = self.fire_module1(X)
        X = self.fire_module2(X)

        return X


class Decoder(nn.Module):

    def __init__(self, out_planes=3):
        super().__init__()

        self.defire_module1 = nn.Sequential(
            DeFire(128, 8, 8, 128),
            DeFire(128, 8, 8, 64),
            DeFire(64, 8, 8, 64),
        )

        self.defire_module2 = nn.Sequential(
            DeFire(64, 8, 8, 32),
            DeFire(32, 8, 8, 32),
        )

        self.gateway = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            nn.ConvTranspose2d(32, out_planes, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, X):
        X = self.defire_module1(X)
        X = self.defire_module2(X)

        return self.gateway(X)


class SqueezeGAN(nn.Module):

    def __init__(self, in_planes=1, out_planes=3):
        super().__init__()
        self.encoder = Encoder(in_planes)

        self.transformer = nn.Sequential(
            nn.Dropout(0.4),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            nn.Dropout(0.1)
        )

        self.decoder = Decoder(out_planes)

        self.final_activation = nn.Sigmoid()

    def forward(self, X):
        X = self.encoder(X)
        X = self.transformer(X)
        X = self.decoder(X)
        return self.final_activation(X)
