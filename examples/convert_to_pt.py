import os

from segtat.convert import convert_geotiff_into_pt

train_features_path = 'D:/segmentation/train_features'
train_label_path = 'D:/segmentation/train_labels'
save_path = 'D:/segmentation/converted'

convert_geotiff_into_pt(train_features_path, train_label_path, save_path)
