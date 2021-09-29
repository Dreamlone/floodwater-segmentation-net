import os
import json
import torch
from functools import partial
import segmentation_models_pytorch as smp
from pathlib import Path
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, space_eval
from predict import neural_network_prediction


INPUT_IMAGES_DIRECTORY = Path("D:/segmentation/train_features")
ASSETS_DIRECTORY = Path("D:/ITMO/floodwater-segmentation-net/codeexecution/assets")


def get_parameters_dict():
    parameters_per_operation = {
        'model_1_th': hp.uniform('model_1_th', 0.0001, 0.5),
        'model_2_th': hp.uniform('model_2_th', 0.0001, 0.5),
        'model_3_th': hp.uniform('model_3_th', 0.0001, 0.5),
        'model_4_th': hp.uniform('model_4_th', 0.0001, 0.5),
        'model_5_th': hp.uniform('model_5_th', 0.0001, 0.5),
        'weight_1': hp.uniform('weight_1', 0.05, 1.0),
        'weight_2': hp.uniform('weight_2', 0.05, 1.0),
        'weight_3': hp.uniform('weight_3', 0.05, 1.0),
        'weight_4': hp.uniform('weight_4', 0.05, 1.0),
        'weight_5': hp.uniform('weight_5', 0.05, 1.0),
    }

    return parameters_per_operation


def read_label(chip_id):
    label_path = 'D:/segmentation/train_labels'
    chip_path = os.path.join(label_path, ''.join((chip_id, '.tif')))
    label_matrix = imread(chip_path)
    return label_matrix


def _objective(parameters_dict, model_1, model_2, model_3, model_4, model_5):
    # Find names of files
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    pr_masks = []
    real_masks = []

    model_1_th = parameters_dict.get('model_1_th')
    model_2_th = parameters_dict.get('model_2_th')
    model_3_th = parameters_dict.get('model_3_th')
    model_4_th = parameters_dict.get('model_4_th')
    model_5_th = parameters_dict.get('model_5_th')
    weight_1 = parameters_dict.get('weight_1')
    weight_2 = parameters_dict.get('weight_2')
    weight_3 = parameters_dict.get('weight_3')
    weight_4 = parameters_dict.get('weight_4')
    weight_5 = parameters_dict.get('weight_5')
    for chip_id in ids:
        pr_mask = neural_network_prediction(chip_id=chip_id, model_1=model_1,
                                            model_2=model_2,
                                            model_3=model_3, model_4=model_4,
                                            model_5=model_5,
                                            model_1_th=model_1_th, model_2_th=model_2_th,
                                            model_3_th=model_3_th,
                                            model_4_th=model_4_th, model_5_th=model_5_th,
                                            weights=[weight_1, weight_2, weight_3, weight_4, weight_5])
        # Real matrix
        real_mask = read_label(chip_id=chip_id)

        # Ignore files with missing pixels
        missing_pixels_number = len(np.argwhere(real_mask == 255))
        if missing_pixels_number == 0:
            pr_masks.append(pr_mask)
            real_masks.append(real_mask)

    pr_tensor = torch.from_numpy(np.array(pr_masks))
    real_tensor = torch.from_numpy(np.array(real_masks))

    # Calculated metric
    iou_metric = smp.utils.metrics.IoU()
    calculated_metric = iou_metric.forward(pr_tensor, real_tensor)

    # We need to maximize metric
    print(f'\nObjective {calculated_metric}')
    print(model_1_th, model_2_th, model_3_th, model_4_th, model_5_th)
    print('Weights', [weight_1, weight_2, weight_3, weight_4, weight_5])
    return -float(calculated_metric)


# Load all neural networks
model_1_path = ASSETS_DIRECTORY / 'julia_net.pth'
model_1 = torch.load(model_1_path).to('cuda')

model_2_path = ASSETS_DIRECTORY / 'manet_19_00_26_09.pth'
model_2 = torch.load(model_2_path).to('cuda')

model_3_path = ASSETS_DIRECTORY / 'fpn_01_00_22_09.pth'
model_3 = torch.load(model_3_path).to('cuda')

model_4_path = ASSETS_DIRECTORY / 'fpn_15_00_26_09.pth'
model_4 = torch.load(model_4_path).to('cuda')

model_5_path = ASSETS_DIRECTORY / 'fpn_14_00_26_09.pth'
model_5 = torch.load(model_5_path).to('cuda')

parameters_dict = get_parameters_dict()

best = fmin(partial(_objective,
                    model_1=model_1,
                    model_2=model_2,
                    model_3=model_3,
                    model_4=model_4,
                    model_5=model_5),
            space=parameters_dict,
            algo=tpe.suggest,
            max_evals=300)

# Get parameters
best = space_eval(space=parameters_dict, hp_assignment=best)

print(best)
with open("D:/segmentation/best_params.json", "w") as fp:
    json.dump(best, fp)
