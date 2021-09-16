import os
import torch
import torch.utils.data as data_utils
import segmentation_models_pytorch as smp

from segtat.metrics import XEDiceLoss


class ModelExplorer:
    """ Class for launching PyTorch Neural networks """

    def __init__(self, working_dir: str, device: bool = 'cpu'):
        self.working_dir = working_dir
        # Loss for all models will be the same
        self.loss = XEDiceLoss()
        self.device = device

    def fit(self, train: torch.tensor, model, **params):
        """ Perform train procedure.
        In **params dict with hyperparameters for neural network can be defined

        :param train: tensor with data for train
        :param model: class PyTorch model
        """
        train_dataset, valid_dataset = self.train_test(train.tensors[0], train.tensors[1], train_size=0.9)
        # Prepare data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

        optimizer = params['optimizer']
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=self.loss,
            metrics=params['metrics'],
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=self.loss,
            metrics=params['metrics'],
            device=self.device,
            verbose=True,
        )

        max_score = 0
        for i in range(0, params['epochs']):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                path_to_save = os.path.join(self.working_dir, 'best_model.pth')
                torch.save(model, path_to_save)
                print('Model saved!')

            # When it takes only 0.7 set
            if i == round(params['epochs']*0.7):
                # Decrease decoder learning rate to 1e-5
                optimizer.param_groups[0]['lr'] = 1e-5

        return model

    def validate(self, test: torch.tensor, model_path: str, model=None, **params):
        """ Perform validation on the test set

        :param test: tensor with test data
        :param model_path: path to the serialised PyTorch model
        :param model: class PyTorch model
        """
        if model is None:
            model = torch.load(os.path.join(self.working_dir, 'best_model.pth'))

        test_loader = torch.utils.data.DataLoader(test)
        test_epoch = smp.utils.train.ValidEpoch(
            model=model,
            loss=self.loss,
            metrics=params['metrics'],
            device=self.device,
        )

        logs = test_epoch.run(test_loader)

    @staticmethod
    def load_data(features_path: str, target_path: str, as_np: bool = False):
        """ Load data from paths

        :param features_path: path to the features .pt file
        :param target_path: path to the label .pt file
        :param as_np: is it needed to return tensors as numpy arrays
        """

        x_train = torch.load(features_path)
        y_train = torch.load(target_path)

        if as_np:
            x_train = x_train.numpy()
            y_train = y_train.numpy()

        return x_train, y_train

    @staticmethod
    def train_test(x_train: torch.tensor, y_train: torch.tensor, train_size: float = 0.8):
        """ Method for train test split

        :param x_train: pytorch tensor with features
        :param y_train: pytorch tensor with labels
        :param train_size: value from 0.1 to 0.9
        """
        if train_size < 0.1 or train_size > 0.9:
            raise ValueError('train_size value must be value between 0.1 and 0.9')
        dataset = data_utils.TensorDataset(x_train, y_train)
        train_ratio = round(len(dataset) * train_size)
        test_ratio = len(dataset) - train_ratio
        train, test = torch.utils.data.random_split(dataset, [train_ratio, test_ratio])

        return train.dataset, test.dataset

    @staticmethod
    def augmentation(dataset: torch.tensor) -> torch.tensor:
        """ Perform augmentation procedure """
        raise NotImplementedError()
