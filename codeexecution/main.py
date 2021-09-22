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

ROOT_DIRECTORY = Path("../codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"


def gauss_filtering(matrix: np.array) -> np.array:
    return gaussian_filter(matrix, sigma=2)


def swi_index(vh_matrix, vv_matrix):
    if len(np.ravel(np.argwhere(vh_matrix == 0))) > 0:
        vh_matrix[vh_matrix == 0] = 0.0001
    if len(np.ravel(np.argwhere(vv_matrix == 0))) > 0:
        vv_matrix[vv_matrix == 0] = 0.0001

    return 0.1747*vv_matrix+0.0082*vh_matrix*vv_matrix+0.0023*np.power(vv_matrix, 2)-0.0015*np.power(vh_matrix, 2)+0.1904


def preprocess_data(arr_vh: np.array, arr_vv: np.array) -> np.array:
    """ Create stacked features numpy tensor """
    scaler_vh_path = ROOT_DIRECTORY / "assets" / 'scaler_vh.pkl'
    scaler_vv_path = ROOT_DIRECTORY / "assets" / 'scaler_vv.pkl'
    scaler_swi_path = ROOT_DIRECTORY / "assets" / 'scaler_swi.pkl'

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

    vh_transformed = scaler_vh.transform(np.ravel(arr_vh).reshape((-1, 1)))
    vh_transformed = vh_transformed.reshape((arr_vh.shape[0], arr_vh.shape[1]))

    vv_transformed = scaler_vv.transform(np.ravel(arr_vv).reshape((-1, 1)))
    vv_transformed = vv_transformed.reshape((arr_vv.shape[0], arr_vv.shape[1]))

    swi_transformed = scaler_swi.transform(np.ravel(arr_swi).reshape((-1, 1)))
    swi_transformed = swi_transformed.reshape((arr_swi.shape[0], arr_swi.shape[1]))

    stacked_matrix = np.array([vh_transformed, vv_transformed, swi_transformed])

    return np.array([stacked_matrix])


def neural_network_prediction(features_matrix: np.array, model):
    """ Make predictions based on FPN neural network """

    features_tensor = torch.from_numpy(features_matrix)
    pr_mask = model.predict(features_tensor.cpu())
    pr_mask = pr_mask.squeeze().cpu().numpy().astype(np.uint8)

    return pr_mask


def make_predictions(chip_id: str, model):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """
    try:
        arr_vh = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif")
        arr_vv = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif")

        # make transformations (gaussian filtering) and scaling
        features = preprocess_data(arr_vh, arr_vv)

        # make predictions
        output_prediction = neural_network_prediction(features, model)
    except:
        logger.warning(
            f"test_features not found for {chip_id}, predicting all zeros; did you download your"
            f"training data into `runtime/data/test_features` so you can test your code?"
        )
        output_prediction = np.zeros(shape=(512, 512), dtype=np.uint8)
    return output_prediction


def get_expected_chip_ids():
    """
    Use the input directory to see which images are expected in the submission
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # images are named something like abc12.tif, we only want the abc12 part
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    """
    for each input file, make a corresponding output file using the `make_predictions` function
    """
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)
    logger.info(f"found {len(chip_ids)} expected image ids; generating predictions for each ...")

    # load neural network
    model_path = ROOT_DIRECTORY / "assets" / 'fpn_01_00_22_09.pth'
    model = torch.load(model_path).cpu()
    for chip_id in tqdm(chip_ids, miniters=25, file=sys.stdout, leave=True):
        # figure out where this prediction data should go
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        # make our predictions
        output_data = make_predictions(chip_id, model)
        imwrite(output_path, output_data, dtype=np.uint8)
    logger.success(f"... done")


if __name__ == "__main__":
    typer.run(main)
