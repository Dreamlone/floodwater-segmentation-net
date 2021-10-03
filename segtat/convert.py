import os
import numpy as np
import pickle
import torch
from geotiff import GeoTiff
from sklearn.preprocessing import StandardScaler
from segtat.preprocessing import rvi_index, swi_index, ratio_index, gauss_filtering


def prepare_simple_scalers(vh_matrix, vv_matrix, save_path):
    """ Prepare and save standard scalers for VH and VV matrices """
    # Train scaler for VH
    scaler_vh = StandardScaler()
    scaler_vh.fit(np.ravel(vh_matrix).reshape((-1, 1)))

    # For VV polarisation
    scaler_vv = StandardScaler()
    scaler_vv.fit(np.ravel(vv_matrix).reshape((-1, 1)))

    # Save scalers into folder
    pickle.dump(scaler_vh, open(os.path.join(save_path, 'scaler_vh_filtered.pkl'), 'wb'))
    pickle.dump(scaler_vv, open(os.path.join(save_path, 'scaler_vv_filtered.pkl'), 'wb'))

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
    pickle.dump(scaler_swi, open(os.path.join(save_path, 'scaler_swi_filtered.pkl'), 'wb'))
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


def _scale_matrix(scaler, current_matrix):
    transformed_matrix = scaler.transform(np.ravel(current_matrix).reshape((-1, 1)))
    transformed_matrix = transformed_matrix.reshape((current_matrix.shape[0], current_matrix.shape[1]))
    return transformed_matrix


def convert_geotiff_into_pt(features_path: str, label_path: str, save_path: str,
                            transformed_indices: bool = False, do_smoothing: bool = False,
                            additional_paths: dict = None):
    """ Method converts geotiff files into npy format

    :param features_path: folder with VV and VH paths
    :param label_path: folder with labeled paths
    :param save_path: folder to save pt files
    :param transformed_indices: is there a need to calculate indices
    :param do_smoothing: is there a need to use gaussian_filter
    :param additional_paths: dictionary with additional layers
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

        if additional_paths is not None:
            # Prepare data from multiple sources
            jrc_change_path = os.path.join(additional_paths['jrc_change'], file)
            jrc_extent_path = os.path.join(additional_paths['jrc_extent'], file)
            jrc_occurrence_path = os.path.join(additional_paths['jrc_occurrence'], file)
            jrc_recurrence_path = os.path.join(additional_paths['jrc_recurrence'], file)
            jrc_seasonality_path = os.path.join(additional_paths['jrc_seasonality'], file)
            jrc_transitions_path = os.path.join(additional_paths['jrc_transitions'], file)
            nasadem_path = os.path.join(additional_paths['nasadem'], file)

            # Read files
            jrc_change_tiff = GeoTiff(jrc_change_path)
            jrc_extent_tiff = GeoTiff(jrc_extent_path)
            jrc_occurrence_tiff = GeoTiff(jrc_occurrence_path)
            jrc_recurrence_tiff = GeoTiff(jrc_recurrence_path)
            jrc_seasonality_tiff = GeoTiff(jrc_seasonality_path)
            jrc_transitions_tiff = GeoTiff(jrc_transitions_path)
            nasadem_tiff = GeoTiff(nasadem_path)

            # Take matrices
            change_matrix = np.array(jrc_change_tiff.read())
            extent_matrix = np.array(jrc_extent_tiff.read())
            occurrence_matrix = np.array(jrc_occurrence_tiff.read())
            recurrence_matrix = np.array(jrc_recurrence_tiff.read())
            seasonality_matrix = np.array(jrc_seasonality_tiff.read())
            transitions_matrix = np.array(jrc_transitions_tiff.read())
            nasadem_matrix = np.array(nasadem_tiff.read())

            # Remove zeros values from matrices
            if len(np.ravel(np.argwhere(vh_matrix == 0))) > 0:
                vh_matrix[vh_matrix == 0] = 0.0001
            if len(np.ravel(np.argwhere(vv_matrix == 0))) > 0:
                vv_matrix[vv_matrix == 0] = 0.0001

            if do_smoothing:
                # Gaussian filter using
                vh_matrix = gauss_filtering(vh_matrix)
                vv_matrix = gauss_filtering(vv_matrix)

            # Calculate SWIX matrix
            swi_matrix = swi_index(vh_matrix, vv_matrix)

            # Take scaling
            if i == 0:
                # Train scaler for VH
                scaler_vh = StandardScaler()
                scaler_vh.fit(np.ravel(vh_matrix).reshape((-1, 1)))

                # Train scaler for VV
                scaler_vv = StandardScaler()
                scaler_vv.fit(np.ravel(vv_matrix).reshape((-1, 1)))

                # Train scaler for change
                scaler_change = StandardScaler()
                scaler_change.fit(np.ravel(change_matrix).reshape((-1, 1)))

                # For extent
                scaler_extent = StandardScaler()
                scaler_extent.fit(np.ravel(extent_matrix).reshape((-1, 1)))

                # For occurrence
                scaler_occurrence = StandardScaler()
                scaler_occurrence.fit(np.ravel(occurrence_matrix).reshape((-1, 1)))

                # For recurrence
                scaler_recurrence = StandardScaler()
                scaler_recurrence.fit(np.ravel(recurrence_matrix).reshape((-1, 1)))

                # For seasonality
                scaler_seasonality = StandardScaler()
                scaler_seasonality.fit(np.ravel(seasonality_matrix).reshape((-1, 1)))

                # For transitions
                scaler_transitions = StandardScaler()
                scaler_transitions.fit(
                    np.ravel(transitions_matrix).reshape((-1, 1)))

                # For nasadem
                scaler_nasadem = StandardScaler()
                scaler_nasadem.fit(np.ravel(nasadem_matrix).reshape((-1, 1)))

                # For SWI index
                scaler_swi = StandardScaler()
                scaler_swi.fit(np.ravel(swi_matrix).reshape((-1, 1)))

                # Save scalers into folder
                pickle.dump(scaler_vh, open(os.path.join(save_path, 'scaler_vh_filtered.pkl'),'wb'))
                pickle.dump(scaler_vv, open(os.path.join(save_path, 'scaler_vv_filtered.pkl'), 'wb'))
                pickle.dump(scaler_change, open(os.path.join(save_path, 'scaler_change.pkl'), 'wb'))
                pickle.dump(scaler_extent, open(os.path.join(save_path, 'scaler_extent.pkl'), 'wb'))
                pickle.dump(scaler_occurrence, open(os.path.join(save_path, 'scaler_occurrence.pkl'), 'wb'))
                pickle.dump(scaler_recurrence, open(os.path.join(save_path, 'scaler_recurrence.pkl'), 'wb'))
                pickle.dump(scaler_seasonality, open(os.path.join(save_path, 'scaler_seasonality_filtered.pkl'), 'wb'))
                pickle.dump(scaler_transitions, open(os.path.join(save_path, 'scaler_transitions.pkl'), 'wb'))
                pickle.dump(scaler_nasadem, open(os.path.join(save_path, 'scaler_nasadem_filtered.pkl'), 'wb'))
                pickle.dump(scaler_swi, open(os.path.join(save_path, 'scaler_swi_filtered.pkl'), 'wb'))

            vh_transformed = _scale_matrix(scaler_vh, vh_matrix)
            vv_transformed = _scale_matrix(scaler_vv, vv_matrix)
            change_transformed = _scale_matrix(scaler_change, change_matrix)
            extent_transformed = _scale_matrix(scaler_extent, extent_matrix)
            occurrence_transformed = _scale_matrix(scaler_occurrence, occurrence_matrix)
            recurrence_transformed = _scale_matrix(scaler_recurrence, recurrence_matrix)
            seasonality_transformed = _scale_matrix(scaler_seasonality, seasonality_matrix)
            transitions_transformed = _scale_matrix(scaler_transitions, transitions_matrix)
            nasadem_transformed = _scale_matrix(scaler_nasadem, nasadem_matrix)
            swi_transformed = _scale_matrix(scaler_swi, swi_matrix)

            # Pack transformed matrices
            stacked_matrix = np.array([vh_transformed, vv_transformed, extent_matrix, seasonality_transformed,
                                       nasadem_transformed, swi_transformed])
        else:
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
                    vh_matrix = gauss_filtering(vh_matrix)
                    vv_matrix = gauss_filtering(vv_matrix)

                # Perform calculations
                rvi_matrix, swi_matrix, ratio_matrix = calculate_indices(
                    vh_matrix,
                    vv_matrix)

                # Train scaler on the first matrices
                if i == 0:
                    scaler_rvi, scaler_swi, scaler_ratio = prepare_advanced_scalers(
                        rvi_matrix,
                        swi_matrix,
                        ratio_matrix,
                        save_path)
                # Perform scaling for indices
                rvi_transformed = scaler_rvi.transform(
                    np.ravel(rvi_matrix).reshape((-1, 1)))
                rvi_transformed = rvi_transformed.reshape(
                    (rvi_matrix.shape[0], rvi_matrix.shape[1]))

                swi_transformed = scaler_swi.transform(
                    np.ravel(swi_matrix).reshape((-1, 1)))
                swi_transformed = swi_transformed.reshape(
                    (swi_matrix.shape[0], swi_matrix.shape[1]))

                ratio_transformed = scaler_ratio.transform(
                    np.ravel(ratio_matrix).reshape((-1, 1)))
                ratio_transformed = ratio_transformed.reshape(
                    (ratio_matrix.shape[0], ratio_matrix.shape[1]))

                # Pack transformed matrices
                stacked_matrix = np.array(
                    [rvi_transformed, swi_transformed, ratio_transformed])
            else:
                if do_smoothing:
                    #########################
                    # Gaussian filter using #
                    #########################
                    vh_matrix = gauss_filtering(vh_matrix)
                    vv_matrix = gauss_filtering(vv_matrix)

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
