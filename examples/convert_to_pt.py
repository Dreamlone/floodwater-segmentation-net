import os

from segtat.convert import convert_geotiff_into_pt

# Simple VH and VV matrices with scaling packing
train_features_path = 'D:/segmentation/train_features'
train_label_path = 'D:/segmentation/train_labels'
save_path = 'D:/segmentation/converted_final'

jrc_change_path = 'D:/segmentation/jrc_change'
jrc_extent_path = 'D:/segmentation/jrc_extent'
jrc_occurrence_path = 'D:/segmentation/jrc_occurrence'
jrc_recurrence_path = 'D:/segmentation/jrc_recurrence'
jrc_seasonality_path = 'D:/segmentation/jrc_seasonality'
jrc_transitions_path = 'D:/segmentation/jrc_transitions'
nasadem_path = 'D:/segmentation/nasadem'
additional_paths = {'jrc_change': jrc_change_path, 'jrc_extent': jrc_extent_path,
                    'jrc_occurrence': jrc_occurrence_path, 'jrc_recurrence': jrc_recurrence_path,
                    'jrc_seasonality': jrc_seasonality_path, 'jrc_transitions': jrc_transitions_path,
                    'nasadem': nasadem_path}

convert_geotiff_into_pt(train_features_path, train_label_path, save_path,
                        do_smoothing=True, additional_paths=additional_paths)

# Advanced fields calculations and packing
# save_path = 'D:/segmentation/converted_ids'
# convert_geotiff_into_pt(train_features_path, train_label_path, save_path,
#                         transformed_indices=True, do_smoothing=True)
