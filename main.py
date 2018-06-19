from squeezegan import SqueezeGAN
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


epochs = 400
# train_size = 9702
train_size = 1
input_max = 64
# res_id = 3000
res_id = 0


def grayify(X):
    Y = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
    return Y.view((Y.size(0), 1, Y.size(1), Y.size(2)))


def criterion(X, Y, Z, discriminator=nn.MSELoss(), distance=nn.MSELoss()):
    return 0.5 (10 * discriminator(X, Y) + 20 * distance(grayify(X), Z)) ** 2


net = SqueezeGAN()
optimizer = optim.Adam(net.parameters())
data = data_iterator()

for epoch in range(epochs):

    running_loss = 0
    for step in range(train_size * 10):
        optimizer.zero_grad()
        inputs, targets, res = next(data)

        outputs = net(inputs)
        loss = criterion(outputs, targets, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if step % 128 == 0:
            # running_loss /= 128
        print(f'epoch : {epoch} step : {step} loss : {loss}')
        running_loss = 0

        if res:
            to_pil = transforms.ToPILImage()
            outputs = outputs[0]
            # outputs *= 255
            # outputs = outputs.view((outputs.size(1), outputs.size(2), outputs.size(0)))
            # outputs = outputs.view((outputs.size(1), outputs.size(2)))
            # outputs = outputs.detach().numpy().astype(np.uint8)
            # im = Image.fromarray(outputs)
            im = to_pil(outputs)
            im.save('res/' + str(epoch) + '.jpg')

    torch.save(net, 'net.pt')
