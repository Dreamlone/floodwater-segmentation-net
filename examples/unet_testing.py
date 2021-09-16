import torch
import os
import torch.utils.data as data_utils
from segtat.preprocessing import test_train_separate
from segtat.convert import convert_geotiff_into_pt

DATA_DIR = 'C:/Users/Julia/Documents/ITMO/NSS_LAB/water_detection/segmentation/'
'''
# separating on test/train
test_train_separate(os.path.join(DATA_DIR, 'train_features'),
                    os.path.join(DATA_DIR, 'train_labels'),
                    os.path.join(DATA_DIR, 'splitted_data'))
'''
# Saving features as test/train torch
train_features_path = os.path.join(DATA_DIR, 'splitted_data/X_train')
train_label_path = os.path.join(DATA_DIR, 'splitted_data/Y_train')
test_features_path = os.path.join(DATA_DIR, 'splitted_data/X_test')
test_label_path = os.path.join(DATA_DIR, 'splitted_data/Y_test')

save_path = os.path.join(DATA_DIR, 'converted_ids/train')
convert_geotiff_into_pt(train_features_path, train_label_path, save_path,
                        transformed_indices=True, do_smoothing=True)
save_path = os.path.join(DATA_DIR, 'converted_ids/test')
convert_geotiff_into_pt(test_features_path, test_label_path, save_path,
                        transformed_indices=True, do_smoothing=True)

# loading data and put it into tensors
train_X = torch.load(os.path.join(DATA_DIR, 'converted_ids/train/X_train.pt'))
train_Y = torch.load(os.path.join(DATA_DIR, 'converted_ids/train/Y_train.pt'))
test_X = torch.load(os.path.join(DATA_DIR, 'converted_ids/test/X_train.pt'))
test_Y = torch.load(os.path.join(DATA_DIR, 'converted_ids/test/Y_train.pt'))

train = data_utils.TensorDataset(train_X, train_Y)
test = data_utils.TensorDataset(test_X, test_Y)

