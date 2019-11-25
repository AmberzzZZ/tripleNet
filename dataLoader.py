import numpy as np
import cv2
import os
import glob
import random
import itertools
from keras.utils import to_categorical


def loadData(data_path, target_size=224, folder=None):
    x_train = []
    y_train = []

    if folder:
        folderlst = [folder]
    else:
        folderlst = [i for i in os.listdir(data_path) if i[0]!='.']
    for idx, folder in enumerate(folderlst):
        for file in glob.glob(os.path.join(data_path, folder)+"/*"):
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (target_size, target_size))
            x_train.append(img)
            y_train.append(idx)

    return np.array(x_train), np.array(y_train)


def generate_triplet(x, y, ap_pairs=10, an_pairs=10):
    triplet_train_pairs = []
    triplet_p_n_labels = []
    data_xy = tuple([x,y])
    for data_class in np.unique(y):
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(itertools.permutations(same_class_idx,2)),k=ap_pairs)
        Neg_idx = random.sample(list(diff_class_idx),k=an_pairs)

        for ap in A_P_pairs:
            try:                                       # might bounds
                Anchor = data_xy[0][ap[0]]
                Positive = data_xy[0][ap[1]]
                for n in Neg_idx:
                    Negative = data_xy[0][n]
                    triplet_train_pairs.append([Anchor,Positive,Negative])
                    triplet_p_n_labels.append([data_class, data_xy[1][n]])
            except:
                continue

    return np.array(triplet_train_pairs), np.array(triplet_p_n_labels)


def triplet_generator(x, y, batch_size, n_classes=10):
    while 1:
        an_pairs = batch_size // 2
        ap_pairs = batch_size - an_pairs
        x_train_pairs, y_train_labels = generate_triplet(x, y, ap_pairs, an_pairs)
        Anchor = np.expand_dims(x_train_pairs[:,0,:], axis=-1)
        Positive = np.expand_dims(x_train_pairs[:,1,:], axis=-1)
        Negative = np.expand_dims(x_train_pairs[:,2,:], axis=-1)
        y_dummy = np.zeros((Anchor.shape[0], 1))
        y_onehot = to_categorical(y_train_labels[:,0], num_classes=n_classes)

        yield [Anchor, Positive, Negative], [y_onehot, y_dummy]


def base_generator(x, y, batch_size, n_classes=10):
    x = np.expand_dims(x, axis=-1)
    idx = [i for i in range(y.shape[0])]
    while 1:
        random.shuffle(idx)
        x_batch = x[idx][:batch_size]
        y_batch = y[idx][:batch_size]
        y_onehot = to_categorical(y_batch, num_classes=n_classes)

        yield x_batch, y_onehot


if __name__ == '__main__':
    data_path = "data/train/"
    batch_size = 4
    n_classes = 3

    x_train, y_train = loadData(data_path, target_size=28)

    for idx, i in enumerate(triplet_generator(x_train, y_train, batch_size, n_classes)):
        [Anchor, Positive, Negative], [y_onehot, y_dummy] = i
        print(y_onehot.shape, y_dummy.shape)
        if idx > 1:
            break










