import inspect
import torch
import torch.nn as nn
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        # self.save_hyperparameters() Equivalent to self.hparams = hparams,
        # this line is equivalent to assigning a value to the self.hparams parameter
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        # self.validation_step_outputs = []  # save outputs of each batch for validation
        # self.test_step_outputs = []  # save outputs of each batch for validation

    # load and init model by model file name and Class name.
    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    # model forward function
    def forward(self, img):
        return self.model(img)

    # todo: define loss function
    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss == 'mse':
            self.loss_function = nn.MSELoss()
        elif loss == 'triplet_margin_loss':
            self.loss_function = nn.TripletMarginLoss()
        else:
            raise ValueError(f'Optimizer {self.loss_function} has not been added to "configure_loss()"')

    # todo: define the training step. This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        img, labels = batch

        # Here we are calling the method forward that we defined above
        out = self(img)
        # Call the loss_function we defined above
        loss = self.loss_function(out, labels)
        # logger=True means that we use tensorboard save loss value
        self.log('loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    # todo: In the case that you need to make use of all the outputs from each training_step(), override the
    #  on_train_epoch_end() method.
    # def on_train_epoch_end(self):
    #     all_preds = torch.stack(self.training_step_outputs)
    #     # do something with all preds
    #     ...
    #     self.training_step_outputs.clear()  # free memory

    # todo: define the validation step.  For validation, we will also iterate step by step over the validation set
    #  when we has multiple validation datasets, we have to set dataloader_idx=None ,this is the way Pytorch Lightning is made.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        out_digit = out.argmax(dim=1)

        correct_num = sum(labels == out_digit).cpu().item()

        # record batch and epoch loss
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, logger=True)
        # record epoch accuracy
        self.log('val_acc', correct_num / len(out_digit), on_epoch=True, prog_bar=True, logger=True)

        # self.validation_step_outputs.append((correct_num, len(out_digit)))
        return (correct_num, len(out_digit))

    # todo: Called in the validation loop at the end of the epoch.
    # def on_validation_epoch_end(self):
    #     # calculate final accuracy of whole epoch
    #     total_correct_num = sum([i[0] for i in self.validation_step_outputs])
    #     total_out_digit = sum([i[1] for i in self.validation_step_outputs])
    #     epoch_accuracy = total_correct_num / total_out_digit
    #     self.log('epoch_val_acc', epoch_accuracy, prog_bar=True, logger=True)
    #
    #     self.validation_step_outputs.clear()  # free memory

    # todo: Operates on a single batch of data from the test set.
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        img, labels = batch
        out = self(img)
        out_digit = out.argmax(dim=1)
        correct_num = sum(labels == out_digit).cpu().item()
        # self.test_step_outputs.append((correct_num, len(out_digit)))
        return (correct_num, len(out_digit))

    # # todo: Called in the test loop at the end of the epoch.
    # def on_test_epoch_end(self):
    #     # calculate final accuracy of whole epoch
    #     total_correct_num = sum([i[0] for i in self.validation_step_outputs])
    #     total_out_digit = sum([i[1] for i in self.validation_step_outputs])
    #     epoch_accuracy = total_correct_num / total_out_digit
    #     self.log('epoch_test_acc', epoch_accuracy, prog_bar=True, logger=True)
    #
    #     self.test_step_outputs.clear()  # free memory

    # todo: Operates on a single batch of data from the predict set.
    def predict_step(self, batch, batch_idx):
        img, labels = batch
        return self(img)

    def configure_optimizers(self):
        # If weight_decay is set, set its value to optimizer
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        # set optimizer and its hparams
        if self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.lr,
                                        weight_decay=weight_decay,
                                        momentum=self.hparams.momentum)

        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=weight_decay)

        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        # Use lr_scheduler
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)

            elif self.hparams.lr_scheduler == 'multi_step':
                scheduler = lrs.MultiStepLR(optimizer,
                                            milestones=self.hparams.milestones,
                                            gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]
