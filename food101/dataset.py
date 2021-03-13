from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import PIL


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
        size_min = min(w,h)
        sw = sh = float(size) / size_min
        
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BICUBIC)


class Food101N(Dataset):
    def __init__(self, split='train', data_path=None, transform=None):
        if data_path is None:
            data_path = 'image_list'

        if split == 'train':
            self.image_list = np.load(os.path.join(data_path, 'train_images.npy'))
            self.targets = np.load(os.path.join(data_path, 'train_targets.npy'))
        else:
            self.image_list = np.load(os.path.join(data_path, 'test_images.npy'))
            self.targets = np.load(os.path.join(data_path, 'test_targets.npy'))

        self.targets = self.targets - 1  # make sure the label is in the range [0, num_class - 1]
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        # image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
        image = resize(image, 256)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label), index

    def __len__(self):
        return len(self.targets)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def gen_train_list():
    root_data_path = '/data/local/pw241/data/Food-101N_release/meta/imagelist.tsv'
    class_list_path = '/data/local/pw241/data/Food-101N_release/meta/classes.txt'

    file_path_prefix = '/data/local/pw241/data/Food-101N_release/images'

    map_name2cat = dict()
    with open(class_list_path) as fp:
        for i, line in enumerate(fp):
            row = line.strip()
            map_name2cat[row] = i
    num_class = len(map_name2cat)
    print('Num Classes: ', num_class)

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        fp.readline()  # skip first line

        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(os.path.join(file_path_prefix, line.strip()))

    targets = np.array(targets)
    img_list = np.array(img_list)
    print('Num Train Images: ', len(img_list))

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'train_images'), img_list)
    np.save(os.path.join(save_dir, 'train_targets'), targets)

    return map_name2cat


def gen_test_list(arg_map_name2cat):
    map_name2cat = arg_map_name2cat
    root_data_path = '/data/local/pw241/data/food-101/meta/test.txt'

    file_path_prefix = '/data/local/pw241/data/food-101/images'

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(os.path.join(file_path_prefix, line.strip() + '.jpg'))

    targets = np.array(targets)
    img_list = np.array(img_list)

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'test_images'), img_list)
    np.save(os.path.join(save_dir, 'test_targets'), targets)

    print('Num Test Images: ', len(img_list))


if __name__ == '__main__':
    map_name2cat = gen_train_list()
    gen_test_list(map_name2cat)
