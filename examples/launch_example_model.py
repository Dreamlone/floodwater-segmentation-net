import segmentation_models_pytorch as smp
import torch

from segtat.explorer import ModelExplorer

"""
Показан пример запсука ModelExplorer'а 
Это вспомогательный инструмент для запуска нейронных сетей и проведения валидации.
Позволяет применять методы аугментации, разделения на обучение и тест.   
"""
explorer = ModelExplorer(working_dir='D:/segmentation', device='cuda')
# Load data as PyTorch tensors
x_train, y_train = explorer.load_data(features_path='D:/segmentation/converted/X_train.pt',
                                      target_path='D:/segmentation/converted/Y_train.pt',
                                      as_np=False)
# Divide into train and test and get Datasets
train, test = explorer.train_test(x_train, y_train, train_size=0.9)

# Augmentation
train = explorer.augmentation(train)

# Initialise Neural network model
nn_model = smp.FPN(encoder_name="resnet18",
                   encoder_weights="imagenet",
                   in_channels=2,
                   classes=1,
                   activation='sigmoid')
optimizer = torch.optim.Adam(params=nn_model.parameters(), lr=0.0001)
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

# Launch network model
fitted_model = explorer.fit(train, nn_model, batch_size=5, epochs=20,
                            optimizer=optimizer, metrics=metrics)

# Validate model on the test dataset
explorer.validate(test, model_path='D:/segmentation/fpn_23_00_17_09.pth', metrics=metrics)
