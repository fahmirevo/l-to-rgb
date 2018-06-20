import torch
import torch.nn as nn
import torch.optim as optim
from models import discriminator
from data import curriculum_iterator
import numpy as np

epochs = 100
train_size = 9702
# train_size = 10
input_max = 64
n_classes = 1
switch_size = 8
# switch_size = 6
display_step = switch_size * 2

net = discriminator.SqueezeNet(3, n_classes)
optimizer = optim.Adam(net.parameters())

criterion = nn.BCELoss()


def switch_generator(switch_size):
    count = -switch_size

    while True:
        count += 1
        if count > switch_size:
            count = -switch_size

        yield count


def noise_lvl(train_size):
    lvl = 1.05
    next_lvl = train_size * 4
    count = 0

    while True:
        count += 1
        if count % next_lvl == 0 and lvl > 0.05:
            next_lvl += train_size * 2
            lvl -= 0.05

        yield lvl


class Augmentor:

    def __init__(self, generator):
        self.generator = generator(train_size, input_max)
        self.noise_lvl = noise_lvl(train_size)
        self.switch = switch_generator(switch_size)

    def __next__(self):
        _, real = next(self.generator)

        if next(self.switch) <= 0:
            target = torch.Tensor([[1]])
            return real, target

        noise_lvl = next(self.noise_lvl)

        noise = np.random.random(real.shape)
        if noise_lvl > 1:
            fake = torch.Tensor(noise)
        else:
            noise[noise > noise_lvl] = 0.5
            noise = 2 * noise - 1
            fake = real + torch.Tensor(noise)
            fake[fake > 1] = 1
            fake[fake < 0] = 0

        target = torch.Tensor([[0]])
        return fake, target


data = Augmentor(curriculum_iterator)

if __name__ == '__main__':
    for epoch in range(epochs):
        running_loss = 0
        for step in range(train_size):
            optimizer.zero_grad()
            inputs, targets = next(data)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print(outputs)
            # print(loss.item())

            # if step % 20 == 19:
            if step % display_step == display_step - 1:
                running_loss /= display_step
                print(f'epoch : {epoch} step : {step} loss : {running_loss}')
                running_loss = 0

    torch.save(net, 'net.pt')

    # for epoch in range(epochs):
    #     running_loss = 0
    #     for step in range(train_size * 10):
    #         optimizer.zero_grad()
    #         inputs, targets = next(data)

    #         outputs = net(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         print(loss.item())

    #         if step % 128 == 0:
    #             running_loss /= 128
    #             print(f'over epoch : {epoch} step : {step} loss : {loss}')
    #             running_loss = 0

    # torch.save(net, 'overnet.pt')
