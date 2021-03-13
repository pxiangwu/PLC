import numpy as np


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_with_P(y_train, nb_classes, noise, random_state=None):

    if noise > 0.0:
        P = build_uniform_P(nb_classes, noise)
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)
        keep_indices = np.arange(len(y_train))

    return y_train, P, keep_indices


def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    keep_indices = np.arange(len(y_train))
    n = noise

    if n > 0.0:
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - n, n

        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - n, n

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - n, n
        P[5, 5], P[5, 3] = 1. - n, n

        # automobile -> truck
        P[4, 4], P[4, 7] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_modelnet40_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        automobile <- truck
        bird -> airplane
        cat <-> dog
        deer -> horse
    """
    nb_classes = 40
    P = np.eye(nb_classes)
    keep_indices = np.arange(len(y_train))
    n = noise

    if n > 0.0:
        # bench -> chair
        P[3, 3], P[3, 9] = 1. - n, n

        # bottle <-> vase
        P[5, 5], P[5, 37] = 1. - n, n
        P[37, 37], P[37, 5] = 1. - n, n

        # desk <-> table
        P[12, 12], P[12, 33] = 1. - n, n
        P[33, 33], P[33, 12] = 1. - n, n

        # flower_pot <-> glass box
        P[15, 15], P[15, 16] = 1. - n, n
        P[16, 16], P[16, 15] = 1. - n, n

        # bowel <-> cup
        P[6, 6], P[6, 10] = 1. - n, n
        P[10, 10], P[10, 6] = 1. - n, n

        # night stand -> table
        P[23, 23], P[23, 33] = 1. - n, n

        # tv stand -> table
        P[36, 36], P[36, 33] = 1. - n, n

        # sofa -> bench
        P[30, 30], P[30, 3] = 1. - n, n

        # bathhub -> sink
        P[1, 1], P[1, 29] = 1. - n, n

        # dresser <-> wardrobe
        P[14, 14], P[14, 38] = 1. - n, n
        P[38, 38], P[38, 14] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    return P


def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    keep_indices = np.arange(len(y_train))

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 <- 7
        2 -> 7
        3 -> 8
        5 <-> 6
    """
    nb_classes = 10
    P = np.eye(nb_classes)
    n = noise

    keep_indices = np.arange(len(y_train))

    if n > 0.0:
        # 1 <- 7
        P[7, 7], P[7, 1] = 1. - n, n

        # 2 -> 7
        P[2, 2], P[2, 7] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 -> 8
        P[3, 3], P[3, 8] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_binary_asymmetric(y_train, noise, random_state=None):
    """mistakes:
        1 -> 0: n
        0 -> 1: .05
    """
    P = np.eye(2)
    n = noise

    keep_indices = np.arange(len(y_train))

    assert 0.0 <= n < 0.5

    if noise > 0.0:
        P[1, 1], P[1, 0] = 1.0 - n, n
        P[0, 0], P[0, 1] = 0.95, 0.05

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    keep_indices = None

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        keep_indices = np.where(y_train_noisy == y_train)[0]

        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

        y_train = y_train_noisy

    return y_train, P, keep_indices


