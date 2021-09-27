import segmentation_models_pytorch as smp
import torch.utils.data as data_utils

from segtat.explorer import ModelExplorer

explorer = ModelExplorer(working_dir='D:/segmentation', device='cuda')
# Load data as PyTorch tensors
x_train, y_train = explorer.load_data(features_path='D:/segmentation/converted_no_missing/X_train.pt',
                                      target_path='D:/segmentation/converted_no_missing/Y_train.pt',
                                      as_np=False)
x_train = x_train.float()
validation = data_utils.TensorDataset(x_train, y_train)
# Validate model on the test dataset
for th in [0.0000000001]:
    print(f'Threshold: {th}')
    metrics = [smp.utils.metrics.IoU(threshold=th)]
    explorer.validate(validation, model_path='D:/segmentation/best_model.pth',
                      metrics=metrics, threshold=th, vis=True)
