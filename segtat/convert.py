import os
import numpy as np
import torch
from geotiff import GeoTiff


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
    for file in label_files:
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

        # TODO apply scaling
        stacked_matrix = np.array([vh_matrix, vv_matrix])
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