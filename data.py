import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def data_iterator(train_size, input_max, monitor=False, monitor_list=None):
    idxs = np.arange(train_size)
    to_tensor = transforms.ToTensor()

    while True:
        for idx in idxs:
            im = Image.open('dataset/' + str(idx) + '.jpg')

            if im.mode != 'RGB':
                continue

            width, height = im.size
            if width > height:
                ratio = input_max / width
                im = im.resize((input_max, int(ratio * height)))
            else:
                ratio = input_max / height
                im = im.resize((int(ratio * width), input_max))

            gray = im.convert('L')

            im = to_tensor(im)
            gray = to_tensor(gray)

            im = im.view(1, im.size(0), im.size(1), im.size(2))
            gray = gray.view(1, gray.size(0), gray.size(1), gray.size(2))

            if not monitor:
                yield gray, im
