import os

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from random import shuffle
from shutil import copyfile


def rvi_index(vh_matrix, vv_matrix):
    return 4*vh_matrix/(vh_matrix+vv_matrix)


def swi_index(vh_matrix, vv_matrix):
    return 0.1747*vv_matrix+0.0082*vh_matrix*vv_matrix+0.0023*np.power(vv_matrix, 2)-0.0015*np.power(vh_matrix, 2)+0.1904


def ratio_index(vh_matrix, vv_matrix):
    return vh_matrix/vv_matrix


def smooth_gaussian_filter(matrix, sigma=5):
    return gaussian_filter(matrix, sigma=sigma)


def normalization(matrix):
    return preprocessing.normalize(matrix, norm='l2')

def test_train_separate(X_train_data_path, Y_train_data_path, out_data_path, test_percent = 0.2):
    Y_train_names = os.listdir(Y_train_data_path)
    shuffle(Y_train_names)
    test_size = int(len(Y_train_names) * test_percent)
    Y_train = Y_train_names[:test_size]
    Y_test = Y_train_names[test_size:]
    X_train = []
    X_test = []
    for file in Y_train:
        X_train.append(f'{file.split(".")[0]}_vv.tif')
        X_train.append(f'{file.split(".")[0]}_vh.tif')
    for file in Y_test:
        X_test.append(f'{file.split(".")[0]}_vv.tif')
        X_test.append(f'{file.split(".")[0]}_vh.tif')
    try:
        os.makedirs(os.path.join(out_data_path, 'X_train'))
        os.makedirs(os.path.join(out_data_path, 'X_test'))
        os.makedirs(os.path.join(out_data_path, 'Y_train'))
        os.makedirs(os.path.join(out_data_path, 'Y_test'))
    except Exception as e:
        print(e)

    for file in Y_train:
        copyfile(os.path.join(Y_train_data_path, file), os.path.join(out_data_path, 'Y_train', file))
    for file in Y_test:
        copyfile(os.path.join(Y_train_data_path, file), os.path.join(out_data_path, 'Y_test', file))
    for file in X_train:
        copyfile(os.path.join(X_train_data_path, file), os.path.join(out_data_path, 'X_train', file))
    for file in X_test:
        copyfile(os.path.join(X_train_data_path, file), os.path.join(out_data_path, 'X_test', file))
