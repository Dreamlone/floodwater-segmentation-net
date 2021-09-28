import os
import numpy as np
from geotiff import GeoTiff

from shutil import copyfile


def filter_data(features_path: str, label_path: str,
                features_save_path: str, label_save_path: str):
    """ Filtering data to prepare only filtered images (with water) """
    # Create folder if it's not exists
    for save_path in [features_save_path, label_save_path]:
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)

    label_files = os.listdir(label_path)
    for file in label_files:
        splitted = file.split('.')
        base_name = splitted[0]
        vh_postfix = '_vh.tif'
        vv_postfix = '_vv.tif'

        file_vh_path = os.path.join(features_path, ''.join((base_name, vh_postfix)))
        file_vv_path = os.path.join(features_path, ''.join((base_name, vv_postfix)))
        file_label_path = os.path.join(label_path, file)

        # Read label matrix
        file_label_tiff = GeoTiff(file_label_path)
        label_matrix = np.array(file_label_tiff.read())

        shape = label_matrix.shape
        water_pixels_number = len(np.argwhere(label_matrix == 1))
        land_pixels_number = len(np.argwhere(label_matrix == 0))
        missing_pixels_number = len(np.argwhere(label_matrix == 255))

        water_ratio = water_pixels_number / (shape[0]*shape[1])
        land_ratio = land_pixels_number / (shape[0] * shape[1])
        missing_ratio = missing_pixels_number / (shape[0] * shape[1])
        print(f'File {base_name} info:')
        print(f'Water ratio: {water_ratio:.2f}, land ratio: {land_ratio:.2f}, '
              f'missing part: {missing_ratio:.2f}')

        if missing_pixels_number == 0:
            # Save label matrix
            new_label_path = os.path.join(label_save_path, file)
            copyfile(file_label_path, new_label_path)

            # Save VH and VV matrices (features)
            new_vh_path = os.path.join(features_save_path, ''.join((base_name, vh_postfix)))
            copyfile(file_vh_path, new_vh_path)

            new_vv_path = os.path.join(features_save_path, ''.join((base_name, vv_postfix)))
            copyfile(file_vv_path, new_vv_path)
        else:
            pass
