""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""

import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from model import MInterface
from data import DInterface


# define callback functions
def load_callbacks():
    callbacks = []

    # use EarlyStopping.
    # The model will stop training after patience epoch where the monitor value (val_acc) is no longer increasing.
    # minimum change in the monitored quantity to qualify as an improvement,
    # i.e. an absolute change of less than or equal to `min_delta`, will count as no improvement.
    # todo: if we use multiple validation datasets, We must specify which dataset corresponds to the indicator being monitored.
    #  Same process should be setted in plc.ModelCheckpoint.
    callbacks.append(plc.EarlyStopping(
        monitor='valid_acc_epoch/dataloader_idx_0',  # todo: change the monitor metric for your dataset
        mode='max',
        patience=10,
        min_delta=0.0001
    ))

    #  the best k models according to the quantity monitored will be saved.
    callbacks.append(plc.ModelCheckpoint(
        # todo: change the monitor metric for your dataset
        monitor='valid_acc_epoch/dataloader_idx_0',
        # todo: change the monitor metric for your dataset
        filename='best_{epoch}_acc_{valid_acc_epoch/dataloader_idx_0:.4f}',
        save_top_k=2,
        mode='max',
        save_last=True,
        save_weights_only=True
    ))

    # Generates a summary of all layers in a LightningModule
    # Note:The Trainer already configured with model summary callbacks by default.
    # callbacks.append(plc.ModelSummary(
    #     max_depth=1
    # ))

    # Automatically monitor and logs learning rate for learning rate schedulers during training.
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))

    return callbacks


def main(args):
    # set random seed
    pl.seed_everything(args.seed)

    # init pytorch_lighting data and model module
    # vars(args) transformer property and value of a python object into a dict
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    # add callbacks to args and send it to Trainer
    args.callbacks = load_callbacks()

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=f'./Logs/{args.model_name}',  # we use current model for log folder name
        max_epochs=args.epochs,
        callbacks=args.callbacks,  # we only run the checkpointing callback (you can add more)
        check_val_every_n_epoch=1,  # run validation every epoch
        log_every_n_steps=20,
        enable_model_summary=True,
        benchmark=True,
        num_sanity_val_steps=0,  # runs a validation step before starting training
        precision=16,  # we use half precision to reduce  memory usage
        # fast_dev_run=True  # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )
    # train and eval model using train_dataloader and eval_dataloader
    trainer.fit(model, data_module)

    # test model using defined test_dataloader, you have to set the ckpt_path
    # trainer.test(model=model, datamodule=data_module,
    #              ckpt_path=r'')


if __name__ == '__main__':
    parser = ArgumentParser()

    # todo: Basic Training Control
    # set random seed
    parser.add_argument('--seed', default=1234, type=int)
    # use GPU or CPU
    parser.add_argument('--accelerator', default='gpu', type=str)
    # select GPU device
    parser.add_argument('--devices', default=[0], type=list)
    # set training epochs
    parser.add_argument('--epochs', default=5, type=int)
    # set batch size
    parser.add_argument('--batch_size', default=32, type=int)
    # set number of process worker in dataloader
    parser.add_argument('--num_workers', default=8, type=int)
    # set init learning rate
    parser.add_argument('--lr', default=1e-3, type=float)
    # select optimizer. We have defined multiple optimizers in model_interface.py, we can select one for our study here.
    parser.add_argument('--optimizer', choices=['sgd', 'adamw', 'adam'], default='sgd', type=str)
    # set momentum of optimizer. It should set for sgd. When we use adam or adamw optimizer, no need to set it
    parser.add_argument('--momentum', default=0.9, type=float)
    # set weight_decay rate for optimizer
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    # todo: LR Scheduler. Used for dynamically adjusting learning rates
    # select lr_scheduler. We have defined multiple lr_scheduler in model_interface.py, we can select one for our study here.
    parser.add_argument('--lr_scheduler', choices=['step', 'multi_step', 'cosine'], default='multi_step', type=str)
    # Here, we can use gradual warmup to , i.e., start with an initially small learning rate,
    # and increase a little bit for each STEP until the initially set relatively large learning rate is reached,
    # and then use the initially set learning rate for training.
    parser.add_argument('--warmup_steps', default=100, type=int)

    # Set args for Different Scheduler
    ## For StepLR
    # parser.add_argument('--lr_decay_steps', default=20, type=int)
    # parser.add_argument('--lr_decay_rate', default=0.5, type=float)

    ## For MultiStepLR
    parser.add_argument('--milestones', default=[5, 10, 20], type=list)
    # lr_decay_rate controls the change rate of learning rate
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)

    ## For CosineAnnealingLR
    # parser.add_argument('--lr_decay_steps', default=20, type=int)
    # parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # todo: Training Info
    # Typically, we need to verify the performance of our model on multiple validation datasets.
    # Here, we can assign train/eval/test datasets. Here, we use standard_data for
    parser.add_argument('--train_dataset', default=['example_traindata'], type=list)
    parser.add_argument('--eval_datasets', default=['example_evaldata', 'example_evaldata'], type=list)
    parser.add_argument('--test_datasets', default=['example_testdata'], type=list)

    # when we have defined multiple models, we can specify which model to use for training here.
    parser.add_argument('--model_name', default='example_net', type=str)

    # select loss function. We have defined multiple loss function in model_interface.py,
    # we can select one or add other loss function here for our study.
    parser.add_argument('--loss', choices=['mse', 'cross_entropy', 'triplet_margin_loss'],
                        default='cross_entropy', type=str)

    # we can use metric functions in TorchMetrics library. we can also  define our own evaluation metric by this libaray.
    # It is friendly to Pytorch Lighting. You can add metric you need to choices and quickly select it.
    parser.add_argument('--metric', choices=['accuracy', 'recall'],
                        default='accuracy', type=str)
    # when our task is multiclass (the --metric is accuracy), we should set how many class for metric in config_metric() function of  MInterface
    parser.add_argument('--num_classes', default=10, type=int)

    # todo: Model Hyperparameters. You need to modify Hyperparameters for your model. Here, it is just an example.
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    args = parser.parse_args()

    main(args)
