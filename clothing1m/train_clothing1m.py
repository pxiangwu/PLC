import sys
sys.path.append('..')

import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import random
import argparse
import os
import numpy as np
from termcolor import cprint
import copy
from tqdm import tqdm
from utils import prob_correction
import logging
import time

from networks.resnet import resnet50
from data_clothing1m import Clothing1M


def main(args):
    np.random.seed(123)

    log_out_dir = 'logs'
    os.makedirs(log_out_dir, exist_ok=True)
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    log_dir = os.path.join(log_out_dir, 'log-' + time_stamp + '.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        filename=log_dir,
                        filemode='w')

    print("Using GPUs:", args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters Setting
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    current_delta: float = 0.7
    flip_threshold = np.ones(args.nepochs) * 0.5
    initial_threshold = np.array([0.8, 0.8, 0.7, 0.6])
    flip_threshold[:len(initial_threshold)] = initial_threshold[:]

    # data augmentation
    transform_train = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = args.data_root
    trainset = Clothing1M(data_root=data_root, split='train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    valset = Clothing1M(data_root=data_root, split='val', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    testset = Clothing1M(data_root=data_root, split='test', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    num_class = 14

    f = resnet50(num_classes=num_class, pretrained=True)
    f = nn.DataParallel(f)
    f.to(device)

    print("\n")
    print("============= Parameter Setting ================")
    print("Using Clothing1M dataset")
    print("Training Epoch : {} | Batch Size : {} | Learning Rate : {} ".format(args.nepochs, batch_size, lr))
    print("================================================")
    print("\n")

    print("============= Start Training =============")
    print("Start Label Correction at Epoch : {}".format(args.warm_up))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[6, 11], gamma=0.5)
    f_record = torch.zeros([args.rollWindow, len(trainset), num_class])

    test_acc = None
    best_val_acc = 0
    best_val_acc_epoch = 0
    best_test_acc = 0
    best_test_acc_epoch = 0
    best_weight = None

    for epoch in range(args.nepochs):
        train_loss = 0
        train_correct = 0
        train_total = 0

        f.train()
        for iteration, (features, labels, indices) in enumerate(tqdm(train_loader, ascii=True, ncols=50)):
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

            # ----------------------------------------------------------------------
            # Evaluation if necessary
            if iteration % args.eval_freq == 0:
                print("\n>> Validation <<")
                f.eval()
                test_loss = 0
                test_correct = 0
                test_total = 0

                for _, (features, labels, indices) in enumerate(val_loader):
                    if features.shape[0] == 1:
                        continue

                    features, labels = features.to(device), labels.to(device)
                    outputs = f(features)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    test_total += features.size(0)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(labels).sum().item()

                val_acc = test_correct / test_total * 100
                cprint(">> [Epoch: {}] Val Acc: {:3.3f}%\n".format(epoch, val_acc), "blue")
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                    best_weight = copy.deepcopy(f.state_dict())
                f.train()

        train_acc = train_correct / train_total * 100
        cprint("Epoch [{}|{}] \t Train Acc {:.3f}%".format(epoch+1, args.nepochs, train_acc), "yellow")
        cprint("Epoch [{}|{}] \t Best Val Acc {:.3f}% \t Best Test Acc {:.3f}%".format(epoch+1, args.nepochs, best_val_acc, best_test_acc), "yellow")
        scheduler.step()

        if epoch >= args.warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets

            y_corrected, current_delta = prob_correction(y_tilde, f_x, random_state=0, thd=0.1, current_delta=current_delta)
            
            logging.info('Current delta:\t{}\n'.format(current_delta))

            trainset.update_corrupted_label(y_corrected)

    # -- Final testing
    f.load_state_dict(best_weight)
    f.eval()

    test_loss = 0
    test_correct = 0
    test_total = 0

    for _, (features, labels, indices) in enumerate(test_loader):
        if features.shape[0] == 1:
            continue

        features, labels = features.to(device), labels.to(device)
        outputs = f(features)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        test_total += features.size(0)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100
    cprint(">> Test Acc: {:3.3f}%\n".format(test_acc), "yellow")
    if best_test_acc < test_acc:
        best_test_acc = test_acc
        best_test_acc_epoch = epoch
    print(">> Best validation accuracy: {:3.3f}%, at epoch {}".format(best_val_acc, best_val_acc_epoch))
    print(">> Best testing accuracy: {:3.3f}%, at epoch {}".format(best_test_acc, best_test_acc_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepochs", default=15, help="number of training epochs", type=int)
    parser.add_argument("--rollWindow", default=5, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--gpus", default='0', help="how many GPUs to use", type=str)
    parser.add_argument("--warm_up", default=1, help="warm-up period", type=int)
    parser.add_argument("--eval_freq", default=2000, help="evaluation frequency (every a few iterations)", type=int)
    parser.add_argument("--top_k", default=3, help="Flip to the top k categories", type=int)
    parser.add_argument("--data_root", default='../../data/Clothing1M/', help="The root to the Clothing1M dataset", type=str)
    args = parser.parse_args()

    main(args)
