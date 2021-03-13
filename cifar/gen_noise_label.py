import os
import sys
sys.path.append('..')
import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import random
import argparse
import copy

import torchvision.transforms as transforms
from cifar_train_val_test import CIFAR10, CIFAR100
from networks.preact_resnet import preact_resnet34
from utils import label_noise, eta_approximation, init_fn_


def main(args):
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters Setting
    batch_size: int = 128
    num_workers: int = 1
    train_val_ratio: float = 0.9
    approximation_args = {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(root='./data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=init_fn_)

        testset = CIFAR10(root='./data', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        num_class = 10
        in_channel = 3

    elif args.dataset == 'cifar100':
        trainset = CIFAR100(root='./data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=init_fn_)

        testset = CIFAR100(root='./data', split='test', download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        num_class = 100
        in_channel = 3
    else:
        raise ValueError('Dataset should be cifar10 or cifar100.')

    n = len(trainset)

    f = preact_resnet34(num_classes=num_class, num_input_channels=in_channel)
    f.to(device)

    # Set the parameters for eta approximation network
    approximation_args['train_loader'] = train_loader
    approximation_args['test_loader'] = test_loader
    approximation_args['f'] = f
    approximation_args['n'] = n
    approximation_args['output_dim'] = num_class
    approximation_args['device'] = device
    approximation_args['n_epochs'] = args.epochs
    approximation_args['lr'] = 1e-3

    print("Modeling Eta")
    eta = eta_approximation(approximation_args)

    print("Synthesizing Noisy Labels ...")
    y_syn = []
    for etaval in eta:
        y_temp = torch.multinomial(etaval, 1)
        y_syn.append(int(y_temp))
    y_syn = np.array(y_syn).squeeze()

    # Final step
    gt_clean_labels = copy.deepcopy(trainset.targets)
    trainset.update_corrupted_label(y_syn.copy())
    y_train_tilde, tau_x = label_noise(trainset, eta, args.noise_type - 1, factor=args.noise_factor)  # in our code, the noise type index starts from 0
    output_noisy_label = y_train_tilde.squeeze().copy()

    # Save the generated noisy labels
    assert len(gt_clean_labels) == len(output_noisy_label), "Size mismatch!"
    real_noise_ratio = np.sum(gt_clean_labels != output_noisy_label) / len(gt_clean_labels)
    print("Real Noise Level: {}".format(real_noise_ratio))

    save_file_name = args.dataset + '-' + str(args.noise_type) + '-' + '%.2f' % real_noise_ratio + '.npy'
    save_file_path = os.path.join('noisy_labels', save_file_name)
    np.save(save_file_path, output_noisy_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_factor", default=1.2, help="constant factor for PMD noise", type=float)
    parser.add_argument("--noise_type", default=1, help="noise type", type=int)
    parser.add_argument("--dataset", default="cifar10", help="experiment dataset", type=str)
    parser.add_argument("--network", default='preact_resnet34', help="network architecture", type=str)
    parser.add_argument("--gpus", default='0', help="which GPU to use", type=str)
    parser.add_argument("--seed", default=123, help="random seed", type=int)
    parser.add_argument("--epochs", default=25, help="number of training epochs", type=int)
    args = parser.parse_args()

    main(args)
