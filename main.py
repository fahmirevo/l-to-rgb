from squeezegan import SqueezeGAN
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from PIL import Image


epochs = 400
train_size = 9702
input_max = 64
res_id = 3000


def data_iterator():
    idxs = np.arange(train_size)

    while True:
        for idx in idxs:
            im = Image.open('dataset/' + str(idx) + '.jpg')

            width, height = im.size
            if width > height:
                ratio = input_max / width
                im = im.resize((input_max, int(ratio * height)))
            else:
                ratio = input_max / height
                im = im.resize((int(ratio * width), input_max))

            gray = im.convert('L')

            im = np.array(im) / 255
            gray = np.array(gray) / 255

            try:
                im = im.reshape((1, 3) + im.shape[:2])
            except Exception as e:
                print(e)
                continue

            gray = gray.reshape((1, 1) + gray.shape)

            im = torch.Tensor(im)
            gray = torch.Tensor(gray)

            if idx == res_id:
                yield gray, im, True
            else:
                yield gray, im, False


def grayify(X):
    Y = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
    return Y.view((Y.size(0), 1, Y.size(1), Y.size(2)))


def criterion(X, Y, Z, discriminator=nn.MSELoss(), distance=nn.MSELoss()):
    return discriminator(X, Y) + 10 * distance(grayify(X), Z)


net = SqueezeGAN()
optimizer = optim.Adam(net.parameters())
data = data_iterator()

for epoch in range(epochs):

    running_loss = 0
    for step in range(train_size):
        optimizer.zero_grad()
        inputs, targets, res = next(data)

        outputs = net(inputs)
        loss = criterion(outputs, targets, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 1 == 0:
            print(f'epoch : {epoch} step : {step} loss : {loss}')
            running_loss = 0

        if res:
            outputs = outputs[0]
            outputs *= 255
            outputs = outputs.view((outputs.size(1), outputs.size(2), outputs.size(0)))
            outputs = outputs.detach().numpy().astype(np.uint8)
            im = Image.fromarray(outputs)
            im.save('res/' + str(epoch) + '.jpg')

    torch.save(net, 'net.pt')
