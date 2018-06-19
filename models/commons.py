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

        self.s1x1_activation = nn.LeakyReLU(inplace=True)
        self.e1x1_activation = nn.LeakyReLU(inplace=True)
        self.e3x3_activation = nn.LeakyReLU(inplace=True)

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

        self.e1x1 = nn.ConvTranspose2d(self.split_planes, e1x1_planes, kernel_size=1)
        self.e3x3 = nn.ConvTranspose2d(self.split_planes, e3x3_planes,
                            kernel_size=3, padding=1)
        self.s1x1 = nn.ConvTranspose2d(e1x1_planes + e3x3_planes,
                            s1x1_planes, kernel_size=1)

        self.e1x1_activation = nn.LeakyReLU(inplace=True)
        self.e3x3_activation = nn.LeakyReLU(inplace=True)
        self.s1x1_activation = nn.LeakyReLU(inplace=True)

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

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, X):
        X = self.conv_block(X) + X
        return self.activation(X)
