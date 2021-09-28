import sys
import pickle
from pathlib import Path

import torch
from loguru import logger
import numpy as np
import typer
from tifffile import imwrite, imread
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


DATA_DIRECTORY = Path("D:/segmentation")
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "train_features"
ASSETS_DIRECTORY = Path("D:/ITMO/floodwater-segmentation-net/codeexecution/assets")


def _scale_matrix(scaler, current_matrix):
    transformed_matrix = scaler.transform(np.ravel(current_matrix).reshape((-1, 1)))
    transformed_matrix = transformed_matrix.reshape((current_matrix.shape[0], current_matrix.shape[1]))
    return transformed_matrix


def gauss_filtering(matrix: np.array) -> np.array:
    return np.array(gaussian_filter(matrix, sigma=2))


def swi_index(vh_matrix, vv_matrix):
    if len(np.ravel(np.argwhere(vh_matrix == 0))) > 0:
        vh_matrix[vh_matrix == 0] = 0.0001
    if len(np.ravel(np.argwhere(vv_matrix == 0))) > 0:
        vv_matrix[vv_matrix == 0] = 0.0001

    return 0.1747*vv_matrix+0.0082*vh_matrix*vv_matrix+0.0023*np.power(vv_matrix, 2)-0.0015*np.power(vh_matrix, 2)+0.1904


def preprocess_filtered_data(chip_id, arr_vh: np.array, arr_vv: np.array) -> np.array:
    extent_data = DATA_DIRECTORY / "jrc_extent"
    seasonality_data = DATA_DIRECTORY / "jrc_seasonality"
    nasadem_data = DATA_DIRECTORY / "nasadem"

    # Read all matrices
    extent_matrix = imread(extent_data / f"{chip_id}.tif")
    extent_matrix = np.array(extent_matrix)

    seasonality_matrix = imread(seasonality_data / f"{chip_id}.tif")
    seasonality_matrix = np.array(seasonality_matrix)

    nasadem_matrix = imread(nasadem_data / f"{chip_id}.tif")
    nasadem_matrix = np.array(nasadem_matrix)

    # Calculate SWI index array
    arr_swi = swi_index(arr_vh, arr_vv)

    ############################################################################
    # S C A L E R S                S C A L E R S                 S C A L E R S #
    ############################################################################
    scaler_vh_path = ASSETS_DIRECTORY / 'scaler_vh_filtered.pkl'
    scaler_vv_path = ASSETS_DIRECTORY / 'scaler_vv_filtered.pkl'
    scaler_swi_path = ASSETS_DIRECTORY / 'scaler_swi_filtered.pkl'
    scaler_nasadem_path = ASSETS_DIRECTORY / 'scaler_nasadem_filtered.pkl'
    scaler_seasonality_path = ASSETS_DIRECTORY / 'scaler_seasonality_filtered.pkl'

    with open(scaler_vh_path, 'rb') as f:
        scaler_vh = pickle.load(f)

    with open(scaler_vv_path, 'rb') as f:
        scaler_vv = pickle.load(f)

    with open(scaler_swi_path, 'rb') as f:
        scaler_swi = pickle.load(f)

    with open(scaler_nasadem_path, 'rb') as f:
        scaler_nasadem = pickle.load(f)

    with open(scaler_seasonality_path, 'rb') as f:
        scaler_seasonality = pickle.load(f)

    vh_transformed = _scale_matrix(scaler_vh, arr_vh)
    vv_transformed = _scale_matrix(scaler_vv, arr_vv)
    swi_transformed = _scale_matrix(scaler_swi, arr_swi)
    nasadem_transformed = _scale_matrix(scaler_nasadem, nasadem_matrix)
    seasonality_transformed = _scale_matrix(scaler_seasonality, seasonality_matrix)

    stacked_matrix = np.array([vh_transformed, vv_transformed, extent_matrix, seasonality_transformed,
                               nasadem_transformed, swi_transformed])

    return np.array([stacked_matrix])


def preprocess_non_filtered_data(arr_vh: np.array, arr_vv: np.array) -> np.array:
    """ Create stacked features numpy tensor """
    scaler_vh_path = ASSETS_DIRECTORY / 'scaler_vh_non_filtered.pkl'
    scaler_vv_path = ASSETS_DIRECTORY/ 'scaler_vv_non_filtered.pkl'
    scaler_swi_path = ASSETS_DIRECTORY / 'scaler_swi_non_filtered.pkl'

    with open(scaler_vh_path, 'rb') as f:
        scaler_vh = pickle.load(f)

    with open(scaler_vv_path, 'rb') as f:
        scaler_vv = pickle.load(f)

    with open(scaler_swi_path, 'rb') as f:
        scaler_swi = pickle.load(f)

    # Apply filtering
    arr_vh = gauss_filtering(np.array(arr_vh))
    arr_vv = gauss_filtering(np.array(arr_vv))
    # Calculate SWI index array
    arr_swi = swi_index(arr_vh, arr_vv)

    vh_transformed = _scale_matrix(scaler_vh, arr_vh)
    vv_transformed = _scale_matrix(scaler_vv, arr_vv)
    swi_transformed = _scale_matrix(scaler_swi, arr_swi)

    stacked_matrix = np.array([vh_transformed, vv_transformed, swi_transformed])

    return np.array([stacked_matrix])


def neural_network_prediction(chip_id: str, model_1, model_2, model_3, model_4, model_5,
                              model_1_th, model_2_th, model_3_th, model_4_th, model_5_th,
                              weights):
    """ Make predictions based on several neural networks """
    arr_vh = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif")
    arr_vv = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif")

    # Convert data into numpy arrays
    arr_vh = np.array(arr_vh)
    arr_vv = np.array(arr_vv)

    # make transformations (gaussian filtering) and scaling
    features_matrix_filtered = preprocess_filtered_data(chip_id, arr_vh, arr_vv)
    features_matrix_non_filtered = preprocess_non_filtered_data(arr_vh, arr_vv)

    features_tensor_filtered = torch.from_numpy(features_matrix_filtered)
    features_tensor_filtered = features_tensor_filtered.float()
    features_tensor_non_filtered = torch.from_numpy(features_matrix_non_filtered)
    # Ensemble predictions
    model_1_mask = model_1.predict(features_tensor_filtered.to('cuda'))
    model_2_mask = model_2.predict(features_tensor_filtered.to('cuda'))
    model_3_mask = model_3.predict(features_tensor_non_filtered.to('cuda'))
    model_4_mask = model_4.predict(features_tensor_filtered.to('cuda'))
    model_5_mask = model_5.predict(features_tensor_non_filtered.to('cuda'))

    model_1_mask = model_1_mask.squeeze().cpu().numpy()
    model_2_mask = model_2_mask.squeeze().cpu().numpy()
    model_3_mask = model_3_mask.squeeze().cpu().numpy()
    model_4_mask = model_4_mask.squeeze().cpu().numpy()
    model_5_mask = model_5_mask.squeeze().cpu().numpy()

    # Binarisation
    model_1_mask[model_1_mask >= model_1_th] = 1
    model_1_mask[model_1_mask < model_1_th] = 0

    model_2_mask[model_2_mask >= model_2_th] = 1
    model_2_mask[model_2_mask < model_2_th] = 0

    model_3_mask[model_3_mask >= model_3_th] = 1
    model_3_mask[model_3_mask < model_3_th] = 0

    model_4_mask[model_4_mask >= model_4_th] = 1
    model_4_mask[model_4_mask < model_4_th] = 0

    model_5_mask[model_5_mask >= model_5_th] = 1
    model_5_mask[model_5_mask < model_5_th] = 0

    stacked_prediction = np.stack([model_1_mask, model_2_mask, model_3_mask,
                                   model_4_mask, model_5_mask])

    pr_mask = np.average(stacked_prediction, axis=0, weights=weights)
    pr_mask[pr_mask >= 0.5] = 1
    pr_mask[pr_mask < 0.5] = 0

    pr_mask = pr_mask.astype(np.uint8)
    return pr_mask
