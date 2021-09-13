import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geotiff import GeoTiff


metadata = pd.read_csv('D:/segmentation/flood-training-metadata.csv')
print(metadata.head(20))

train_features_path = 'D:/segmentation/train_features'
train_label_path = 'D:/segmentation/train_labels'

label_files = os.listdir(train_label_path)
# Download all files
for i, file in enumerate(label_files):
    splitted = file.split('.')
    base_name = splitted[0]
    vh_postfix = '_vh.tif'
    vv_postfix = '_vv.tif'

    file_vh_path = os.path.join(train_features_path, ''.join((base_name, vh_postfix)))
    file_vv_path = os.path.join(train_features_path, ''.join((base_name, vv_postfix)))
    label_path = os.path.join(train_label_path, file)

    # Good example of water and other
    if i == 50:
        # Read geotiff files
        file_vh_tiff = GeoTiff(file_vh_path)
        file_vv_tiff = GeoTiff(file_vv_path)
        file_label_tiff = GeoTiff(label_path)

        # Read as arrays
        vh_matrix = np.array(file_vh_tiff.read())
        vv_matrix = np.array(file_vv_tiff.read())
        label_matrix = np.array(file_label_tiff.read())

        # Visualisation
        plt.imshow(vh_matrix, cmap='jet')
        plt.title('VH matrix')
        plt.colorbar()
        plt.show()

        plt.imshow(vv_matrix, cmap='jet')
        plt.title('VV matrix')
        plt.colorbar()
        plt.show()

        plt.imshow(label_matrix, cmap='Blues')
        plt.title('Label matrix')
        plt.colorbar()
        plt.show()
