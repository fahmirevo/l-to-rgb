import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, in_planes, s1x1_planes,
                 e1x1_planes, e3x3_planes):
        super().__init__()

        self.s1x1 = nn.Conv2d(in_planes, s1x1_planes, kernel_size=1)
        self.e1x1 = nn.Conv2d(s1x1_planes, e1x1_planes, kernel_size=1)
        self.e3x3 = nn.Conv2d(s1x1_planes, e3x3_planes,
                              kernel_size=3, padding=1)

        self.s1x1_activation = nn.ReLU(inplace=True)
        self.e1x1_activation = nn.ReLU(inplace=True)
        self.e3x3_activation = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.s1x1_activation(self.s1x1(X))

        return torch.cat([
            self.e1x1_activation(self.e1x1(X)),
            self.e3x3_activation(self.e3x3(X)),
        ], dim=1)


class DeFire(nn.Module):

    def __init__(self, in_planes, e1x1_planes,
                 e3x3_planes, s1x1_planes):
        super().__init__()

        self.split_planes = in_planes // 2
        # self.e1x1_planes = e1x1_planes
        # self.e3x3_planes = e3x3_planes

        self.e1x1 = nn.ConvTranspose2d(self.split_planes, e1x1_planes, kernel_size=1)
        self.e3x3 = nn.ConvTranspose2d(self.split_planes, e3x3_planes,
                            kernel_size=3, padding=1)
        self.s1x1 = nn.ConvTranspose2d(e1x1_planes + e3x3_planes,
                            s1x1_planes, kernel_size=1)

        self.e1x1_activation = nn.ReLU(inplace=True)
        self.e3x3_activation = nn.ReLU(inplace=True)
        self.s1x1_activation = nn.ReLU(inplace=True)

    def forward(self, X):
        res1 = self.e1x1_activation(self.e1x1(X[:, :self.split_planes]))
        res2 = self.e3x3_activation(self.e3x3(X[:, self.split_planes:]))

        res = torch.cat([res1, res2], dim=1)

        X = self.s1x1(res)
        return self.s1x1_activation(X)


class ResBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_planes, in_planes,
                      kernel_size=3, padding=1),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=3, padding=1),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.conv_block(X) + X
        return self.activation(X)


class Encoder(nn.Module):

    def __init__(self, in_planes=1):
        super().__init__()
        self.gateway = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1,
                                  ceil_mode=True, return_indices=True)

        self.fire_module1 = nn.Sequential(
            Fire(32, 16, 32, 32),
            Fire(64, 16, 32, 32),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1,
                                  ceil_mode=True, return_indices=True)

        self.fire_module2 = nn.Sequential(
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 16, 64, 64),
        )

        self.pool3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1,
                                  ceil_mode=True, return_indices=True)

    def forward(self, X):
        X = self.gateway(X)
        X, idx1 = self.pool1(X)
        X = self.fire_module1(X)
        X, idx2 = self.pool2(X)
        X = self.fire_module2(X)
        X, idx3 = self.pool3(X)

        return X, idx1, idx2, idx3


class Decoder(nn.Module):

    def __init__(self, out_planes=3):
        super().__init__()

        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=1, padding=1)

        self.defire_module1 = nn.Sequential(
            DeFire(128, 8, 8, 128),
            DeFire(128, 8, 8, 64),
            DeFire(64, 8, 8, 64),
        )

        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=1, padding=1)

        self.defire_module2 = nn.Sequential(
            DeFire(64, 8, 8, 32),
            DeFire(32, 8, 8, 32),
        )

        self.unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=1, padding=1)

        self.gateway = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            nn.ConvTranspose2d(32, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, X, idx1, idx2, idx3):
        X = self.unpool1(X, idx3)
        X = self.defire_module1(X)
        X = self.unpool2(X, idx2)
        X = self.defire_module2(X)
        X = self.unpool3(X, idx1)

        return self.gateway(X)


class SqueezeGAN(nn.Module):

    def __init__(self, in_planes=1, out_planes=3):
        super().__init__()
        self.encoder = Encoder(in_planes)

        self.transformer = nn.Sequential(
            nn.Dropout(0.5),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            nn.Dropout(0.5)
        )

        self.decoder = Decoder()

        self.final_activation = nn.Sigmoid()

    def forward(self, X):
        X, idx1, idx2, idx3 = self.encoder(X)
        X = self.transformer(X)
        X = self.decoder(X, idx1, idx2, idx3)
        return self.final_activation(X)
