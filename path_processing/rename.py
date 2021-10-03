import os
from shutil import copyfile

path = 'D:/segmentation/train_features'

files = os.listdir(path)
for file in files:
    if file.endswith('change.tif'):
        path_to_save = 'D:/segmentation/jrc_change'

    elif file.endswith('extent.tif'):
        path_to_save = 'D:/segmentation/jrc_extent'

    elif file.endswith('occurrence.tif'):
        path_to_save = 'D:/segmentation/jrc_occurrence'

    elif file.endswith('recurrence.tif'):
        path_to_save = 'D:/segmentation/jrc_recurrence'

    elif file.endswith('seasonality.tif'):
        path_to_save = 'D:/segmentation/jrc_seasonality'

    elif file.endswith('transitions.tif'):
        path_to_save = 'D:/segmentation/jrc_transitions'

    elif file.endswith('nasadem.tif'):
        path_to_save = 'D:/segmentation/nasadem'

    splitted = file.split('_jrc')[0]
    new_file_name = ''.join((splitted, '.tif'))
    new_path = os.path.join(path_to_save, new_file_name)

    file_path = os.path.join(path, file)
    copyfile(file_path, new_path)
