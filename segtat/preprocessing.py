import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing


def rvi_index(vh_matrix, vv_matrix):
    return 4*vh_matrix/(vh_matrix+vv_matrix)


def swi_index(vh_matrix, vv_matrix):
    return 0.1747*vv_matrix+0.0082*vh_matrix*vv_matrix+0.0023*np.power(vv_matrix, 2)-0.0015*np.power(vh_matrix, 2)+0.1904


def ratio_index(vh_matrix, vv_matrix):
    return vh_matrix/vv_matrix


def gauss_filtering(matrix):
    return gaussian_filter(matrix, sigma=2)


def normalization(matrix):
    return preprocessing.normalize(matrix, norm='l2')
