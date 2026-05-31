"""训练入口。

main.py 只负责三件事：解析命令行参数、创建 DataModule/LightningModule、
创建 Trainer。模型和数据集的具体实现放在 model/ 与 data/ 中。
"""

from __future__ import annotations

from argparse import ArgumentParser

import lightning as L
from data import DInterface
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from model import MInterface


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Clean PyTorch Lightning template")

    # 训练控制参数：默认值必须能在 CPU 上直接跑通。
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--fast_dev_run", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--experiment_name", default="example_net")
    parser.add_argument("--enable_progress_bar", nargs="?", const=True, default=False, type=str_to_bool)

    # 数据参数：val/test 支持多个数据集名称。
    parser.add_argument("--train_dataset", default="example_data")
    parser.add_argument("--val_datasets", nargs="+", default=["example_data"])
    parser.add_argument("--test_datasets", nargs="+", default=["example_data"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_samples", default=128, type=int)
    parser.add_argument("--image_size", default=28, type=int)

    # 模型参数。
    parser.add_argument("--model_name", default="example_net")
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_classes", default=10, type=int)

    # 优化器、损失和指标参数。
    parser.add_argument("--loss", choices=["cross_entropy", "mse", "triplet_margin_loss"], default="cross_entropy")
    parser.add_argument("--metric", choices=["accuracy", "recall"], default="accuracy")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adam")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--lr_scheduler", choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--lr_decay_steps", default=10, type=int)
    parser.add_argument("--lr_decay_rate", default=0.5, type=float)
    parser.add_argument("--lr_decay_min_lr", default=1e-5, type=float)
    return parser


def build_callbacks(args):
    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor="val_accuracy/dataloader_idx_0",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}",
            auto_insert_metric_name=False,
        ),
    ]
    if args.lr_scheduler != "none":
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def build_trainer(args) -> L.Trainer:
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)
    return L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
        callbacks=build_callbacks(args),
        log_every_n_steps=1,
        enable_progress_bar=args.enable_progress_bar,
    )


def main(args=None) -> None:
    parser = build_parser()
    args = parser.parse_args(args=args)
    L.seed_everything(args.seed, workers=True)

    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))
    trainer = build_trainer(args)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
