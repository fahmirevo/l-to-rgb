import torch
import torch.nn as nn
import torch.optim as optim
from models import discriminator
from data import data_iterator
import numpy as np

epochs = 400
train_size = 9702
input_max = 64
n_classes = 1

net = discriminator.SqueezeNet(3, n_classes)
optimizer = optim.Adam(net.parameters())

if n_classes > 1:
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()


class Augmentor:

    def __init__(self, generator):
        self.generator = generator(train_size, input_max)

    def __next__(self):
        _, real = next(self.generator)

        is_real = np.random.random() < 0.5
        if is_real:
            target = torch.Tensor([[1]])
            return real, target

        noise_lvl = np.random.random()
        noise = np.random.random(real.shape)
        if noise_lvl < 0.1:
            noise **= 3
            mask = np.random.random(real.shape) < np.random.random()
            noise[mask] = 0
        elif noise_lvl < 0.3:
            noise **= 3
        elif noise_lvl < 0.6:
            noise **= 2
        else:
            pass

        fake = real + torch.Tensor(noise)
        fake[fake > 1] = 1

        target = torch.Tensor([[0]])
        return fake, target


data = Augmentor(data_iterator)


for epoch in range(epochs):
    running_loss = 0
    for step in range(train_size * 2):
        optimizer.zero_grad()
        inputs, targets = next(data)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 128 == 0:
            running_loss /= 128
            print(f'epoch : {epoch} step : {step} loss : {running_loss}')
            running_loss = 0

torch.save(net, 'net.pt')

for epoch in range(epochs):
    running_loss = 0
    for step in range(train_size * 10):
        optimizer.zero_grad()
        inputs, targets = next(data)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 128 == 0:
            running_loss /= 128
            print(f'over epoch : {epoch} step : {step} loss : {loss}')
            running_loss = 0

torch.save(net, 'overnet.pt')
