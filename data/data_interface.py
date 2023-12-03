import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# Here, we define the interface inherit from pl.LightningDataModule.
# We can control the batch size, train/eval/test datasets used in our study by args from main.py,
# because all args can be input to DInterface by **kwargs in __init__ function.
class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_workers = kwargs['num_workers']

        # get train/eval/test dataset name list from kwargs
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_datasets']
        self.test_dataset = kwargs['test_datasets']

        # get batch size from kwargs
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs

    def load_data_module(self, dataset_name):
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in dataset_name.split('_')])
        try:
            data_module = getattr(importlib.import_module(
                '.' + dataset_name, package=__package__), camel_name)
            return data_module
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset_name}.{camel_name}')

    def instancialize(self, data_module, **other_args):
        """ Instancialize a data Class using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return data_module(**args1)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # init train dataset
            self.train_set = []
            for item in self.train_dataset:
                self.train_set.append(self.instancialize(self.load_data_module(item)))

            # init eval datasets
            self.eval_set = []
            for item in self.eval_dataset:
                self.eval_set.append(self.instancialize(self.load_data_module(item)))

        # Assign test dataset for use in dataloader(s)
        # you can put multiple dataset into a list and return list for test
        if stage == 'test' or stage is None:
            # init test datasets
            self.test_set = []
            for item in self.test_dataset:
                self.test_set.append(self.instancialize(self.load_data_module(item)))

    def train_dataloader(self):
        # if there is only one train dataset, return its DataLoader
        if len(self.train_set) == 1:
            return DataLoader(self.train_set[0], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                              persistent_workers=True)
        # if there are multiple train dataset, return a DataLoader list
        else:
            train_dataloaders = []
            for dataset in self.train_set:
                train_dataloaders.append(DataLoader(
                    dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True))
            return train_dataloaders

    def val_dataloader(self):
        # if there is only one eval dataset, return its DataLoader
        if len(self.eval_set) == 1:
            return DataLoader(self.eval_set[0], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                              persistent_workers=True)
        # if there are multiple eval dataset, return a DataLoader list
        else:
            eval_dataloaders = []
            for dataset in self.eval_set:
                eval_dataloaders.append(DataLoader(
                    dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                    persistent_workers=True))
            return eval_dataloaders

    def test_dataloader(self):
        # if there is only one test dataset, return its DataLoader
        if len(self.eval_set) == 1:
            return DataLoader(self.test_set[0], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                              persistent_workers=True)
        # if there are multiple test dataset, return a DataLoader list
        else:
            test_dataloaders = []
            for dataset in self.test_set:
                test_dataloaders.append(DataLoader(
                    dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False))
            return test_dataloaders
