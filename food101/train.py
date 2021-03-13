import sys
sys.path.append('..')
import torch.nn as nn
import torch.nn.parallel
import random
import argparse
from networks.resnet import resnet50
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset import Food101N
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from termcolor import cprint
import copy
import time
from utils import lrt_correction
import logging


# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


def main(args):
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    print('Random Seed {}\n'.format(random_seed))

    # -- training parameters
    num_epoch = args.epoch
    milestone = [10, 20]
    batch_size = args.batch
    num_workers = 4

    weight_decay = 1e-3
    gamma = 0.1
    current_delta = args.delta

    lr = args.lr
    start_epoch = 0

    # -- specify dataset
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(224),
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
    trainset = Food101N(data_path=data_root, split='train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              worker_init_fn=_init_fn, drop_last=True)

    testset = Food101N(data_path=data_root, split='test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    num_class = 101

    print('train data size:', len(trainset))
    print('test data size:', len(testset))

    # -- create log file
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = 'Ours(' + time_stamp + ').txt'

    log_dir = 'food101_logs'
    os.makedirs('food101_logs', exist_ok=True)
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")

    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    # -- set network, optimizer, scheduler, etc
    net = resnet50(num_classes=num_class, pretrained=True)
    net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    # -- misc
    iterations = 0
    f_record = torch.zeros([args.rollWindow, len(trainset), num_class])

    for epoch in range(start_epoch, num_epoch):
        train_correct = 0
        train_loss = 0
        train_total = 0

        net.train()

        for i, (images, labels, indices) in enumerate(trainloader):
            if images.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                continue

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            f_record[epoch % args.rollWindow, indices] = F.softmax(outputs.detach().cpu(), dim=1)

            iterations += 1
            if iterations % 100 == 0:
                cur_train_acc = train_correct / train_total * 100.
                cur_train_loss = train_loss / train_total
                cprint('epoch: {}\titerations: {}\tcurrent train accuracy: {:.4f}\ttrain loss:{:.4f}'.format(
                    epoch, iterations, cur_train_acc, cur_train_loss), 'yellow')

                if iterations % 5000 == 0:
                    saver.write('epoch: {}\titerations: {}\ttrain accuracy: {}\ttrain loss: {}\n'.format(
                        epoch, iterations, cur_train_acc, cur_train_loss))
                    saver.flush()
            
            if iterations % args.eval_freq == 0:
                net.eval()
                test_total = 0
                test_correct = 0
                with torch.no_grad():
                    for i, (images, labels, _) in enumerate(testloader):
                        images, labels = images.to(device), labels.to(device)

                        outputs = net(images)

                        test_total += images.size(0)
                        _, predicted = outputs.max(1)
                        test_correct += predicted.eq(labels).sum().item()

                    test_acc = test_correct / test_total * 100.
                    
                cprint('>> Test accuracy: {:.4f}'.format(test_acc), 'cyan')
                saver.write('>> Test accuracy: {}\n'.format(test_acc))
                saver.flush()
                net.train()

        train_acc = train_correct / train_total * 100.

        cprint('epoch: {}'.format(epoch), 'yellow')
        cprint('train accuracy: {:.4f}\ntrain loss: {:.4f}'.format(train_acc, train_loss), 'yellow')
        saver.write('epoch: {}\ntrain accuracy: {}\ntrain loss: {}\n'.format(epoch, train_acc, train_loss))
        saver.flush()

        exp_lr_scheduler.step()

        if epoch >= args.warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets

            y_corrected, current_delta = lrt_correction(y_tilde, f_x, current_delta=current_delta, delta_increment=0.05)

            saver.write('Current delta:\t{}\n'.format(current_delta))

            trainset.update_corrupted_label(y_corrected)

    saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1', help='delimited list input of GPUs', type=str)
    parser.add_argument("--warm_up", default=4, help="warm-up period", type=int)
    parser.add_argument("--rollWindow", default=4, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--eval_freq", default=2000, help="evaluation frequency (every a few iterations)", type=int)
    parser.add_argument("--batch", default=32, help="batch size", type=int)
    parser.add_argument("--epoch", default=30, help="total number of epochs", type=int)
    parser.add_argument("--seed", default=11, help="random seed", type=int)
    parser.add_argument("--lr", default=0.002, help="learning rate", type=float)
    parser.add_argument("--delta", default=0.1, help="initial threshold in Algorithm 1", type=float)
    parser.add_argument("--data_root", default='image_list', help="The root to the dataset", type=str)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    main(args)
