from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import PIL
from os import listdir


# https://github.com/kuangliu/pytorch-retinanet/blob/master/transform.py
def resize(img, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        sw = sh = float(size) / size_min

        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow, oh), Image.BICUBIC)


class Animal10(Dataset):
    def __init__(self, split='train', data_path=None, transform=None):
        if data_path is None:
            data_path = '/home/pwu/Downloads/animal10'

        self.image_dir = os.path.join(data_path, split + 'ing')

        self.image_files = [f for f in listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]

        self.targets = []

        for path in self.image_files:
            label = path.split('_')[0]
            self.targets.append(int(label))

        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])

        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]


if __name__ == '__main__':
    d = Animal10('test')
    print(len(d))

