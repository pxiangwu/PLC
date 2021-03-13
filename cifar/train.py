import sys
sys.path.append('..')

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.nn as nn

import random
import os
import copy
import argparse
from tqdm import tqdm
import numpy as np
from cifar_train_val_test import CIFAR10, CIFAR100
from termcolor import cprint

from utils import lrt_correction
from networks.preact_resnet import preact_resnet34
from noise import noisify_with_P, noisify_cifar10_asymmetric, \
    noisify_cifar100_asymmetric, noisify_mnist_asymmetric, noisify_pairflip


# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


# @profile
def main(args):
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    noise_label_path = os.path.join('noisy_labels', args.noise_label_file)
    noise_y = np.load(noise_label_path)
    print('Load noisy label from {}'.format(noise_label_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters Setting
    batch_size: int = 128
    num_workers: int = 2
    train_val_ratio: float = 0.9
    lr: float = args.lr
    current_delta: float = args.delta

    which_data_set = args.noise_label_file.split('-')[0]
    noise_level = args.noise_level
    noise_type = args.noise_type  # "uniform", "asymmetric"

    # data_ augmentation
    if which_data_set[:5] == 'cifar':
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
    else:
        raise ValueError('Dataset should be cifar10 or cifar100.')

    if which_data_set == 'cifar10':
        trainset = CIFAR10(root='data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn)

        valset = CIFAR10(root='data', split='val', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = CIFAR10(root='data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 10
        in_channel = 3
    elif which_data_set == 'cifar100':
        trainset = CIFAR100(root='data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn)

        valset = CIFAR100(root='data', split='val', train_ratio=train_val_ratio, trust_ratio=0, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = CIFAR100(root='data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 100
        in_channel = 3
    else:
        raise ValueError('Dataset should be cifar10 or cifar100.')

    print('train data size:', len(trainset))
    print('validation data size:', len(valset))
    print('test data size:', len(testset))

    # -- Sanity Check --
    num_noise_class = len(np.unique(noise_y))
    assert num_noise_class == num_class, "The data class number between the generate noisy label and the selected dataset is incorrect!"
    assert len(noise_y) == len(trainset), "The number of noisy label is inconsistent with the training data number!"

    # -- generate noise --
    gt_clean_y = copy.deepcopy(trainset.get_data_labels())
    y_train = noise_y.copy()

    noise_y_train = None
    p = None

    if noise_type == "uniform":
        noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_class, noise=noise_level,
                                                        random_state=random_seed)
        trainset.update_corrupted_label(noise_y_train)
        print("apply uniform noise")
    else:
        if which_data_set == 'cifar10':
            noise_y_train, p, _ = noisify_cifar10_asymmetric(y_train, noise=noise_level, random_state=random_seed)
        elif which_data_set == 'cifar100':
            noise_y_train, p, _ = noisify_cifar100_asymmetric(y_train, noise=noise_level, random_state=random_seed)
        else:
            raise ValueError('Dataset should be cifar10 or cifar100.') 

        trainset.update_corrupted_label(noise_y_train)
        print("apply asymmetric noise")
    print("probability transition matrix:\n{}\n".format(p))

    real_noise_level = np.sum(noise_y_train != gt_clean_y) / len(noise_y_train)
    print('\n>> Real Noise Level: {}'.format(real_noise_level))
    y_train_tilde = copy.deepcopy(noise_y_train)
    y_syn = copy.deepcopy(noise_y_train)

    # -- create log file
    file_name = '(' + which_data_set + ')' + 'Ours_type-' + \
                noise_type + '_noise-' + str(noise_level) + '.txt'
    log_dir = 'log/logs_txt_' + str(random_seed)
    os.makedirs(log_dir, exist_ok=True)
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")
    saver.write('noise level: {}\nnoise type: {}\n'.format(noise_level, noise_type))
    saver.write('transition matrix:\n{}'.format(p))

    # -- set network, optimizer, scheduler, etc
    f = preact_resnet34(num_input_channels=in_channel, num_classes=num_class)

    print("\n")
    print("============= Parameter Setting ================")
    print("Using Data Set : {}".format(which_data_set))
    print("Training Epoch : {} | Batch Size : {} | Learning Rate : {} ".format(args.nepoch, batch_size, lr))
    print("================================================")
    print("\n")

    print("============= Start Training =============")
    print("-- Start Label Correction at Epoch : {} --\n".format(args.warm_up))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)
    f = f.to(device)

    f_record = torch.zeros([args.rollWindow, len(y_train_tilde), num_class])

    best_acc = 0
    best_epoch = 0
    best_weights = None

    for epoch in range(args.nepoch):
        train_loss = 0
        train_correct = 0
        train_total = 0

        f.train()
        for _, (features, labels, _, indices) in enumerate(tqdm(trainloader, ascii=True, ncols=50)):
            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = f(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += features.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            f_record[epoch % args.rollWindow, indices] = F.softmax(outputs.detach().cpu(), dim=1)

        train_acc = train_correct / train_total * 100
        cprint("Epoch [{}|{}] \t Train Acc {:.3f}%".format(epoch+1, args.nepoch, train_acc), "yellow")

        if epoch >= args.warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets
            y_corrected, current_delta = lrt_correction(np.array(y_tilde).copy(), f_x, current_delta=current_delta, delta_increment=args.inc)

            trainset.update_corrupted_label(y_corrected)

        # --- validation --
        val_total = 0
        val_correct = 0
        f.eval()
        with torch.no_grad():
            for i, (images, labels, _, _) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device)

                outputs = f(images)

                val_total += images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total * 100.

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(f.state_dict())

        cprint('val accuracy: {}'.format(val_acc), 'cyan')
        cprint('>> best accuracy: {}\n>> best epoch: {}\n'.format(best_acc, best_epoch), 'green')
        saver.write('Val Accuracy: {}\n'.format(val_acc))

        scheduler.step()

    # -- Final testing
    cprint('>> testing using best validation model <<', 'cyan')
    test_total = 0
    test_correct = 0

    f.load_state_dict(best_weights)
    f.eval()

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = f(images)

            test_total += images.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100.
    cprint('>> test accuracy: {}'.format(test_acc), 'cyan')
    saver.write('>> Final test accuracy: {}\n'.format(test_acc))

    print("\nFinal Clean Ratio {:.3f}%".
          format(sum(np.array(trainset.targets).flatten() == np.array(y_syn).flatten())/float(len(np.array(y_syn)))*100))

    print("Final Test Accuracy {:3.3f}%".format(test_acc))
    print("Final Delta Used {}".format(current_delta))
    return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_label_file', default='cifar10-1-0.35.npy', help='noise label file', type=str)
    parser.add_argument('--noise_type', default='uniform', help='noise type [uniform | asym]', type=str)
    parser.add_argument('--noise_level', default=0.0, help='noise level [for additional uniform/asymmetric noise applied to the PMD noise]', type=float)

    parser.add_argument("--delta", default=0.3, help="initial threshold", type=float)
    parser.add_argument("--nepoch", default=180, help="number of training epochs", type=int)
    parser.add_argument("--rollWindow", default=5, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--gpus", default=0, help="which GPU to use", type=int)
    parser.add_argument("--warm_up", default=8, help="warm-up period", type=int)
    parser.add_argument("--seed", default=77, help="random seed", type=int)
    parser.add_argument("--lr", default=0.01, help="initial learning rate", type=float)
    parser.add_argument("--inc", default=0.1, help="increment", type=float)
    args = parser.parse_args()
    main(args)
