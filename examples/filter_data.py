import os

from segtat.filtering import filter_data

features_path = 'D:/segmentation/train_features'
label_path = 'D:/segmentation/train_labels'
features_save_path = 'D:/segmentation/no_missing_features'
label_save_path = 'D:/segmentation/no_missing_labels'
filter_data(features_path, label_path,
            features_save_path, label_save_path)
