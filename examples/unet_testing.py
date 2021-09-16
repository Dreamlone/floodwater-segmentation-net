import torch
import os
import torch.utils.data as data_utils
import segmentation_models_pytorch as smp
from segtat.preprocessing import test_train_separate
from segtat.convert import convert_geotiff_into_pt

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

DEVICE = get_device()
print(f'Calculations on device: {DEVICE}')

DATA_DIR = 'C:/Users/Julia/Documents/ITMO/NSS_LAB/water_detection/segmentation/'
'''
# separating on test/train
test_train_separate(os.path.join(DATA_DIR, 'train_features'),
                    os.path.join(DATA_DIR, 'train_labels'),
                    os.path.join(DATA_DIR, 'splitted_data'))

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
'''
# loading data and put it into tensors
train_X = torch.load(os.path.join(DATA_DIR, 'converted_ids/train/X_train.pt'))
train_Y = torch.load(os.path.join(DATA_DIR, 'converted_ids/train/Y_train.pt'))
test_X = torch.load(os.path.join(DATA_DIR, 'converted_ids/test/X_train.pt'))
test_Y = torch.load(os.path.join(DATA_DIR, 'converted_ids/test/Y_train.pt'))

train = data_utils.TensorDataset(train_X, train_Y)
test = data_utils.TensorDataset(test_X, test_Y)

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=0)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, num_workers=0)

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=5,
    classes=2,                      # model output channels (number of classes in your dataset)
)

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# create epoch runners
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for epochs
epochs = 10
max_score = 0
for i in range(0, epochs):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(test_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, os.path.join(DATA_DIR, 'converted_ids/best_model.pth'))
        print('Model saved!')

    if i == int(epochs/2):
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
