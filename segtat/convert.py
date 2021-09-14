import os
import numpy as np
import pickle
import torch
from geotiff import GeoTiff
from sklearn.preprocessing import StandardScaler
from segtat.preprocessing import rvi_index, swi_index, ratio_index, gaussian_filter


def prepare_simple_scalers(vh_matrix, vv_matrix, save_path):
    """ Prepare and save standard scalers for VH and VV matrices """
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


def prepare_advanced_scalers(rvi_matrix, swi_matrix, ratio_matrix, save_path):
    """ Prepare and save standard scalers for RVI, SWI and ratio matrices """
    # Train scaler for RVI
    scaler_rvi = StandardScaler()
    scaler_rvi.fit(np.ravel(rvi_matrix).reshape((-1, 1)))

    # For SWI
    scaler_swi = StandardScaler()
    scaler_swi.fit(np.ravel(swi_matrix).reshape((-1, 1)))

    # For ratio
    scaler_ratio = StandardScaler()
    scaler_ratio.fit(np.ravel(ratio_matrix).reshape((-1, 1)))

    # Save scalers into folder
    pickle.dump(scaler_rvi, open(os.path.join(save_path, 'scaler_rvi.pkl'), 'wb'))
    pickle.dump(scaler_swi, open(os.path.join(save_path, 'scaler_swi.pkl'), 'wb'))
    pickle.dump(scaler_ratio, open(os.path.join(save_path, 'scaler_ratio.pkl'), 'wb'))

    return scaler_rvi, scaler_swi, scaler_ratio


def fix_label_matrix(label_matrix):
    """ Remove 255 values (NoDATA or OutOfExtend) from matrix """
    no_data_ids = np.argwhere(label_matrix == 255)
    if len(np.ravel(no_data_ids)) != 0:
        # All OutOfExtend pixels becomes 'land'
        label_matrix[label_matrix == 255] = 0
        return label_matrix
    else:
        return label_matrix


def load_matrices(features_path, label_path, file):
    """ Function load matrices in numpy format """
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

    return vh_matrix, vv_matrix, label_matrix


def calculate_indices(vh_matrix, vv_matrix):
    """ Calculate new fields from VH and VV matrices """
    rvi_matrix = rvi_index(vh_matrix, vv_matrix)
    swi_matrix = swi_index(vh_matrix, vv_matrix)
    ratio_matrix = ratio_index(vh_matrix, vv_matrix)
    return rvi_matrix, swi_matrix, ratio_matrix


def convert_geotiff_into_pt(features_path: str, label_path: str, save_path: str,
                            transformed_indices: bool = False, do_smoothing: bool = False):
    """ Method converts geotiff files into npy format

    :param features_path: folder with VV and VH paths
    :param label_path: folder with labeled paths
    :param save_path: folder to save pt files
    :param transformed_indices: is there a need to calculate indices
    :param do_smoothing: is there a need to use gaussian_filter
    """
    # Create folder if it's not exists
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)

    features_save = 'X_train.pt'
    target_save = 'Y_train.pt'
    label_files = os.listdir(label_path)

    features_tensor = []
    target_tensor = []
    for i, file in enumerate(label_files):
        # Load VH, VV and label matrix
        vh_matrix, vv_matrix, label_matrix = load_matrices(features_path, label_path, file)

        # Fix values in label matrices
        label_matrix = fix_label_matrix(label_matrix)

        if transformed_indices:
            # Remove zeros values from matrices
            if len(np.ravel(np.argwhere(vh_matrix == 0))) > 0:
                vh_matrix[vh_matrix == 0] = 0.0001
            if len(np.ravel(np.argwhere(vv_matrix == 0))) > 0:
                vv_matrix[vv_matrix == 0] = 0.0001

            if do_smoothing:
                #########################
                # Gaussian filter using #
                #########################
                vh_matrix = gaussian_filter(vh_matrix)
                vv_matrix = gaussian_filter(vv_matrix)

            # Perform calculations
            rvi_matrix, swi_matrix, ratio_matrix = calculate_indices(vh_matrix, vv_matrix)

            # Train scaler on the first matrices
            if i == 0:
                scaler_rvi, scaler_swi, scaler_ratio = prepare_advanced_scalers(rvi_matrix,
                                                                                swi_matrix,
                                                                                ratio_matrix,
                                                                                save_path)
            # Perform scaling for VH and VV
            rvi_transformed = scaler_rvi.transform(np.ravel(rvi_matrix).reshape((-1, 1)))
            rvi_transformed = rvi_transformed.reshape((rvi_matrix.shape[0], rvi_matrix.shape[1]))

            swi_transformed = scaler_swi.transform(np.ravel(swi_matrix).reshape((-1, 1)))
            swi_transformed = swi_transformed.reshape((swi_matrix.shape[0], swi_matrix.shape[1]))

            ratio_transformed = scaler_ratio.transform(np.ravel(ratio_matrix).reshape((-1, 1)))
            ratio_transformed = ratio_transformed.reshape((ratio_matrix.shape[0], ratio_matrix.shape[1]))

            # Pack transformed matrices
            stacked_matrix = np.array([rvi_transformed, swi_transformed, ratio_transformed])
        else:
            if do_smoothing:
                #########################
                # Gaussian filter using #
                #########################
                vh_matrix = gaussian_filter(vh_matrix)
                vv_matrix = gaussian_filter(vv_matrix)

            # Train scaler on the first matrices
            if i == 0:
                scaler_vh, scaler_vv = prepare_simple_scalers(vh_matrix, vv_matrix, save_path)

            # Perform scaling for VH and VV
            vh_transformed = scaler_vh.transform(np.ravel(vh_matrix).reshape((-1, 1)))
            vh_transformed = vh_transformed.reshape((vh_matrix.shape[0], vh_matrix.shape[1]))
            vv_transformed = scaler_vv.transform(np.ravel(vv_matrix).reshape((-1, 1)))
            vv_transformed = vv_transformed.reshape((vv_matrix.shape[0], vv_matrix.shape[1]))

            # Pack transformed VH and VV matrices
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
