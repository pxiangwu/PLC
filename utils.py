import os
import os.path
import hashlib
import errno
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F
from termcolor import cprint

import numpy as np
from scipy.special import softmax

def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under. If None, use the basename of the URL
        md5 (str): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
                )


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# random seed related
def init_fn_(worker_id):
    np.random.seed(77 + worker_id)


def label_noise(dataset, eta, type, factor=1.2):

    count = 0
    y_train = np.array(torch.tensor(dataset.targets).clone())
    classes = len(np.unique(y_train))

    if classes == 2:
        eta_u = eta[:, 1]
        if type == 0:
            f_us = 2*eta_u*(eta_u-1/2)**2
        if type == 1:
            f_us = (eta_u >= 1/2)*(1-eta_u) + (eta_u < 1/2)*eta_u
        if type == 2:
            f_us = -2*(eta_u-1/2)**2 + 1/2

        for i in range(len(y_train)):
            if y_train[i] == 1:
                y_train[i] = torch.tensor(np.random.binomial(1, 1 - f_us[i], 1), dtype=torch.long)
                if y_train[i] == 0:
                    count += 1

    if classes > 2 and type == 0:
        print(">> Using type-I noise <<")
        temp = eta.topk(2)
        eta_u = np.array(temp[0][:, 0])
        eta_s = np.array(temp[0][:, 1])
        u = np.array(temp[1][:, 0])
        s = np.array(temp[1][:, 1])
        f_us = -(1/2)*(eta_u-eta_s)**2 + 1/2

        for i in range(len(y_train)):
            noise_level = np.maximum((1-f_us), 0.5)
            noise_ind = np.random.binomial(1, noise_level[i]/factor, 1)
            y_train[i] = noise_ind*u[i] + (1-noise_ind)*s[i]
            if not y_train[i] == dataset.targets[i]:
                count += 1

    if classes > 2 and type == 1:
        print(">> Using type-II noise <<")
        temp = eta.topk(2)
        eta_u = np.array(temp[0][:, 0])
        eta_s = np.array(temp[0][:, 1])
        u = np.array(temp[1][:, 0])
        s = np.array(temp[1][:, 1])
        f_us = 1-np.abs(eta_u-eta_s)**3

        for i in range(len(y_train)):
            noise_level = 1-f_us
            noise_ind = np.random.binomial(1, noise_level[i]/factor, 1)
            y_train[i] = noise_ind*u[i] + (1-noise_ind)*s[i]
            if not y_train[i] == dataset.targets[i]:
                count += 1

    if classes > 2 and type == 2:
        print(">> Using type-III noise <<")
        temp = eta.topk(2)
        eta_u = np.array(temp[0][:, 0])
        eta_s = np.array(temp[0][:, 1])
        u = np.array(temp[1][:, 0])
        s = np.array(temp[1][:, 1])
        f_us = 1 - (1/3)*np.abs(eta_u-eta_s)**3 - (1/3)*np.abs(eta_u-eta_s)**2 - (1/3)*np.abs(eta_u-eta_s)

        for i in range(len(y_train)):
            noise_level = 1-f_us
            noise_ind = np.random.binomial(1, noise_level[i]/factor, 1)
            y_train[i] = noise_ind*u[i] + (1-noise_ind)*s[i]
            if not y_train[i] == dataset.targets[i]:
                count += 1

    print("Corrupted Size {} | Noisy Level {:.3f}%".format(count, count/float(len(y_train))*100))

    return y_train, f_us


def eta_approximation(approximation_args):
    # initialization
    train_loader = approximation_args['train_loader']
    test_loader = approximation_args['test_loader']
    f = approximation_args['f']
    n = approximation_args['n']
    output_dim = approximation_args['output_dim']
    device = approximation_args['device']
    n_epochs = approximation_args['n_epochs']
    lr = approximation_args['lr']

    eta = torch.zeros([n, output_dim])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)

    for epoch in range(n_epochs):
        train_loss = 0
        test_loss = 0
        train_correct = 0
        test_correct = 0
        train_total = 0
        test_total = 0

        f.train()
        for _, (features, labels, softlabels, indices) in enumerate(train_loader):
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

            if epoch == n_epochs-1:
                eta[indices] = F.softmax(outputs.detach().cpu(), dim=1)

        train_acc = train_correct / train_total * 100
        print("Epoch [{}|{}] \t Train Acc {:.3f}".format(epoch+1, n_epochs, train_acc))

        if epoch == n_epochs-1:
            f.eval()
            for _, (features, labels, softlabels, indices) in enumerate(test_loader):
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
            print("Final Test Acc {:.3f}".format(test_acc))

    return eta


def lrt_correction(y_tilde, f_x, current_delta=0.3, delta_increment=0.1):
    """
    Label correction using likelihood ratio test. 
    In effect, it gradually decreases the threshold according to Algorithm 1.
    
    current_delta: The initial threshold $\theta$
    delta_increment: The step size, corresponding to the $\beta$ in Algorithm 1.
    """
    corrected_count = 0
    y_noise = torch.tensor(y_tilde).clone()
    n = len(y_noise)
    f_m = f_x.max(1)[0]
    y_mle = f_x.argmax(1)
    LR = []
    for i in range(len(y_noise)):
        LR.append(float(f_x[i][int(y_noise[i])]/f_m[i]))

    for i in range(int(len(y_noise))):
        if LR[i] < current_delta:
            y_noise[i] = y_mle[i]
            corrected_count += 1

    if corrected_count < 0.001*n:
        current_delta += delta_increment
        current_delta = min(current_delta, 0.9)
        cprint("Update Critical Value -> {}".format(current_delta), "red")

    return y_noise, current_delta


def prob_correction(y_noise, f_x, random_state=0, current_delta=0.3, delta_increment=0.1, thd=0.1):
    """
    Correct the noisy labels in a probabilistic manner.

    f_x: numpy array, [data_num, category_num]
    current_delta: The initial threshold $\theta$
    delta_increment: The step size, corresponding to the $\beta$ in Algorithm 1.
    thd: confidence threshold. If the predicted confidence exceeds this threshold, we use label correction based on likelihood ratio test.
        Otherwise, we use probabilistic label correction. 
    """
    flipper = np.random.RandomState(random_state)
    f_x = f_x.cpu().numpy()

    correction_count = 0

    for i in range(f_x.shape[0]):
        cur_prob_distri = f_x[i]
        cur_prob_distri = softmax(cur_prob_distri)

        top_k_idx = np.argsort(cur_prob_distri)[-1:]
        top_probs = cur_prob_distri[top_k_idx]

        if top_probs[-1] >= thd:  # only flip to the category with max probability
            if cur_prob_distri[y_noise[i]]/top_probs[-1] < current_delta:
                new_label = top_k_idx[-1]
                correction_count +=1

                y_noise[i] = new_label
        else:
            top_probs = top_probs / np.sum(top_probs)  # normalization
            flipped = flipper.multinomial(1, top_probs, 1)[0]
            new_label = np.where(flipped == 1)[0]
            new_label = top_k_idx[new_label[0]]  # new_label shape [1, ]

            y_noise[i] = new_label

    if not correction_count:
        current_delta += delta_increment

    return y_noise, current_delta
