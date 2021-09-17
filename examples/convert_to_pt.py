import os

from segtat.convert import convert_geotiff_into_pt

# Simple VH and VV matrices with scaling packing
train_features_path = 'D:/segmentation/train_features'
train_label_path = 'D:/segmentation/train_labels'
save_path = 'D:/segmentation/converted'
convert_geotiff_into_pt(train_features_path, train_label_path, save_path,
                        do_smoothing=True)

# Advanced fields calculations and packing
# save_path = 'D:/segmentation/converted_ids'
# convert_geotiff_into_pt(train_features_path, train_label_path, save_path,
#                         transformed_indices=True)
