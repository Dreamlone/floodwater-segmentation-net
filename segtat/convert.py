import os
import numpy as np
import pickle
import torch
from geotiff import GeoTiff
from sklearn.preprocessing import StandardScaler


def prepare_scalers(vh_matrix, vv_matrix, save_path):
    """ Prepare and save standard scalers for matrices """
    # Train scaler for VH
    scaler_vh = StandardScaler()
    scaler_vh.fit(np.ravel(vh_matrix).reshape((-1, 1)))

    # For VV polarisation
    scaler_vv = StandardScaler()
    scaler_vv.fit(np.ravel(vv_matrix).reshape((-1, 1)))

    # Save scalers into folder
    pickle.dump(scaler_vh, open(os.path.join(save_path, 'scaler_vh.pkl'), 'wb'))
    pickle.dump(scaler_vv, open(os.path.join(save_path, 'scaler_vv.pkl'), 'wb'))

    return scaler_vh, scaler_vv


def fix_label_matrix(label_matrix):
    """ Remove 255 values (NoDATA or OutOfExtend) from matrix """
    no_data_ids = np.argwhere(label_matrix == 255)
    if len(np.ravel(no_data_ids)) != 0:
        # All OutOfExtend pixels becomes 'land'
        label_matrix[label_matrix == 255] = 0
        return label_matrix
    else:
        return label_matrix


def convert_geotiff_into_pt(features_path: str, label_path: str, save_path: str):
    """ Method converts geotiff files into npy format

    :param features_path: folder with VV and VH paths
    :param label_path: folder with labeled paths
    :param save_path: folder to save pt files
    """
    features_save = 'X_train.pt'
    target_save = 'Y_train.pt'
    label_files = os.listdir(label_path)

    features_tensor = []
    target_tensor = []
    for i, file in enumerate(label_files):
        splitted = file.split('.')
        base_name = splitted[0]
        vh_postfix = '_vh.tif'
        vv_postfix = '_vv.tif'

        file_vh_path = os.path.join(features_path, ''.join((base_name, vh_postfix)))
        file_vv_path = os.path.join(features_path, ''.join((base_name, vv_postfix)))
        file_label_path = os.path.join(label_path, file)

        # Read geotiff files
        file_vh_tiff = GeoTiff(file_vh_path)
        file_vv_tiff = GeoTiff(file_vv_path)
        file_label_tiff = GeoTiff(file_label_path)

        # Read as arrays
        vh_matrix = np.array(file_vh_tiff.read())
        vv_matrix = np.array(file_vv_tiff.read())
        label_matrix = np.array(file_label_tiff.read())

        # Fix values in label matrices
        label_matrix = fix_label_matrix(label_matrix)

        if i == 0:
            scaler_vh, scaler_vv = prepare_scalers(vh_matrix, vv_matrix, save_path)

        # Perform scaling
        vh_transformed = scaler_vh.transform(np.ravel(vh_matrix).reshape((-1, 1)))
        vh_transformed = vh_transformed.reshape((vh_matrix.shape[0], vh_matrix.shape[1]))
        vv_transformed = scaler_vv.transform(np.ravel(vv_matrix).reshape((-1, 1)))
        vv_transformed = vv_transformed.reshape((vv_matrix.shape[0], vv_matrix.shape[1]))

        # Pack transformed matrices
        stacked_matrix = np.array([vh_transformed, vv_transformed])
        features_tensor.append(stacked_matrix)
        target_tensor.append([label_matrix])

    features_tensor = np.array(features_tensor)
    target_tensor = np.array(target_tensor)

    # Numpy arrays into tensors
    torch_features_tensor = torch.from_numpy(features_tensor)
    torch_target_tensor = torch.from_numpy(target_tensor)

    features_save_path = os.path.join(save_path, features_save)
    target_save_path = os.path.join(save_path, target_save)

    torch.save(torch_features_tensor, features_save_path)
    torch.save(torch_target_tensor, target_save_path)
