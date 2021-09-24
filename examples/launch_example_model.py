import segmentation_models_pytorch as smp
import torch

from segtat.explorer import ModelExplorer

"""
Показан пример запсука ModelExplorer'а 
Это вспомогательный инструмент для запуска нейронных сетей и проведения валидации.
Позволяет применять методы аугментации, разделения на обучение и тест.   
"""
explorer = ModelExplorer(working_dir='D:/segmentation', device='cpu')
# Load data as PyTorch tensors
x_train, y_train = explorer.load_data(features_path='D:/segmentation/converted_final/X_train.pt',
                                      target_path='D:/segmentation/converted_final/Y_train.pt',
                                      as_np=False)
x_train = x_train.float()

# Divide into train and test and get Datasets
train, test = explorer.train_test(x_train, y_train, train_size=0.9)

# Augmentation
train = explorer.augmentation(train)

# Initialise Neural network model
nn_model = smp.Unet(encoder_name="resnet18",
                    encoder_weights="imagenet",
                    in_channels=8,
                    classes=1,
                    activation='sigmoid')
optimizer = torch.optim.Adam(params=nn_model.parameters(), lr=0.0001)
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

# Launch network model
fitted_model = explorer.fit(train, nn_model, batch_size=10, epochs=50,
                            optimizer=optimizer, metrics=metrics)

# Validate model on the test dataset
explorer.validate(test, model_path='D:/segmentation/best_model.pth', metrics=metrics)
