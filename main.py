from models.squeezegan import SqueezeGAN
from models.discriminator import SqueezeNet
from data import data_iterator
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


epochs = 400
train_size = 9702
input_max = 64
monitor_list = [3000, 3923, 3548, 3544, 2734, 2434, 1434, 8149, 8359]


def train_d(discriminator, inputs, targets, optimizer, criterion=nn.BCELoss()):
    optimizer.zero_grad()
    outputs = discriminator(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return outputs


def train_g(generator, outputs, d_pred, optimizer, criterion=nn.BCELoss()):
    optimizer.zero_grad()
    loss = criterion(d_pred, torch.Tensor[[0, 1]])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(loss.items())


g = SqueezeGAN()
g_optim = optim.Adam(g.parameters())

d = SqueezeNet(3, 3)
d_optim = optim.Adam(d.parameters())

data = data_iterator(train_size, input_max,
                     monitor=True, monitor_list=monitor_list)

for epoch in range(epochs):

    running_loss = 0
    for step in range(train_size * 10):
        g_optim.zero_grad()
        d_optim.zero_grad()
        inputs, reals, res = next(data)

        train_d(d, reals, torch.Tensor([[1, 0]]), d_optim)

        generated = g(inputs)

        d_pred = train_d(d, generated, torch.Tensor([[0, 1]]), d_optim)

        train_g(g, generated, d_pred, g_optim)

        if res:
            to_pil = transforms.ToPILImage()
            generated = generated[0]
            im = to_pil(generated)
            im.save('res/' + str(epoch) + str(step) + '.jpg')
