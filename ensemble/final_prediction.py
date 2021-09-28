import os
import torch
import torch.utils.data as data_utils
import segmentation_models_pytorch as smp
from pathlib import Path
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt

from predict import neural_network_prediction

INPUT_IMAGES_DIRECTORY = Path("D:/segmentation/train_features")
ASSETS_DIRECTORY = Path("D:/ITMO/floodwater-segmentation-net/codeexecution/assets")


def read_label(chip_id):
    label_path = 'D:/segmentation/train_labels'
    chip_path = os.path.join(label_path, ''.join((chip_id, '.tif')))
    label_matrix = imread(chip_path)
    return label_matrix


def validate_on_train_dataset(vis=False):
    # load neural networks
    model_1_path = ASSETS_DIRECTORY / 'unet_21_00_26_09.pth'
    model_1 = torch.load(model_1_path).to('cuda')

    model_2_path = ASSETS_DIRECTORY / 'manet_19_00_26_09.pth'
    model_2 = torch.load(model_2_path).to('cuda')

    model_3_path = ASSETS_DIRECTORY / 'fpn_01_00_22_09.pth'
    model_3 = torch.load(model_3_path).to('cuda')

    model_4_path = ASSETS_DIRECTORY / 'fpn_15_00_26_09.pth'
    model_4 = torch.load(model_4_path).to('cuda')

    model_5_path = ASSETS_DIRECTORY / 'fpn_14_00_26_09.pth'
    model_5 = torch.load(model_5_path).to('cuda')

    # Find names of files
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    pr_masks = []
    real_masks = []
    for chip_id in ids:
        print(f'Processing {chip_id} file ...')
        # TODO change parameters to founded
        pr_mask = neural_network_prediction(chip_id=chip_id, model_1=model_1, model_2=model_2,
                                            model_3=model_3, model_4=model_4, model_5=model_5,
                                            model_1_th=0.4786118845013668,
                                            model_2_th=0.45231541148089927,
                                            model_3_th=0.012616133131346047,
                                            model_4_th=0.09113130377469408,
                                            model_5_th=0.2565778485812052,
                                            weights=[0.07785132604881947, 0.8577876324711068, 0.6313899058700289, 0.1651720604375764, 0.600658681775319])
        # Real matrix
        real_mask = read_label(chip_id=chip_id)

        # Ignore files with missing pixels
        missing_pixels_number = len(np.argwhere(real_mask == 255))
        if missing_pixels_number == 0:
            pr_masks.append(pr_mask)
            real_masks.append(real_mask)

            if vis is True:
                plt.imshow(real_mask, cmap='Blues', alpha=0.5)
                plt.imshow(pr_mask, cmap='Blues', alpha=0.5)
                plt.colorbar()
                plt.show()

    pr_tensor = torch.from_numpy(np.array(pr_masks))
    real_tensor = torch.from_numpy(np.array(real_masks))

    loss = smp.utils.losses.JaccardLoss()
    calculated_loss = loss.forward(pr_tensor, real_tensor)

    # Calculated metric
    iou_metric = smp.utils.metrics.IoU()
    calculated_metric = iou_metric.forward(pr_tensor, real_tensor)

    return float(calculated_loss), float(calculated_metric)


calculated_loss, calculated_metric = validate_on_train_dataset(vis=True)
print(f'Obtained results for train set: {calculated_loss:.4f}, {calculated_metric:.4f}')
