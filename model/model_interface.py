"""模型模块统一入口。

MInterface 封装 LightningModule 的通用训练逻辑。新增模型时，只需要在
model/ 下添加普通 nn.Module，再通过 --model_name 指定即可。
"""

import importlib
import inspect

import lightning as L
import torch
from torch import nn
from torch.optim import lr_scheduler as schedulers
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall


class MInterface(L.LightningModule):
    """通用 LightningModule，负责模型、损失、指标和优化器配置。"""

    def __init__(
        self,
        model_name: str = "example_net",
        loss: str = "cross_entropy",
        metric: str = "accuracy",
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        lr_scheduler: str = "none",
        lr_decay_steps: int = 10,
        lr_decay_rate: float = 0.5,
        lr_decay_min_lr: float = 1e-5,
        num_classes: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = self._load_model(model_name, kwargs)
        self.loss_function = self._build_loss(loss)
        self.metric_name = metric
        self.valid_metric = self._build_metric(metric, num_classes)
        self.test_metric = self._build_metric(metric, num_classes)

    def _load_model(self, model_name: str, extra_kwargs: dict):
        class_name = "".join(part.capitalize() for part in model_name.split("_"))
        try:
            module = importlib.import_module(f".{model_name}", package=__package__)
            model_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"无法加载模型 model/{model_name}.py 中的 {class_name} 类。") from exc

        signature = inspect.signature(model_cls.__init__)
        accepted = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        model_kwargs = {name: extra_kwargs[name] for name in accepted if name in extra_kwargs}
        model_kwargs.update(
            {
                name: getattr(self.hparams, name)
                for name in accepted
                if hasattr(self.hparams, name) and name not in model_kwargs
            }
        )
        return model_cls(**model_kwargs)

    @staticmethod
    def _build_loss(loss: str):
        loss = loss.lower()
        if loss == "cross_entropy":
            return nn.CrossEntropyLoss()
        if loss == "mse":
            return nn.MSELoss()
        if loss == "triplet_margin_loss":
            return nn.TripletMarginLoss()
        raise ValueError(f"未支持的损失函数: {loss}")

    @staticmethod
    def _build_metric(metric: str, num_classes: int):
        metric = metric.lower()
        if metric == "accuracy":
            return MulticlassAccuracy(num_classes=num_classes)
        if metric == "recall":
            return MulticlassRecall(num_classes=num_classes)
        raise ValueError(f"未支持的评价指标: {metric}")

    def forward(self, images):
        return self.model(images)

    def _shared_eval_step(self, batch, stage: str):
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        metric = self.valid_metric if stage == "val" else self.test_metric
        metric.update(logits, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=images.size(0))
        self.log(f"{stage}_{self.metric_name}", metric, prog_bar=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        return self(images).softmax(dim=1)

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer.lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"未支持的优化器: {optimizer_name}")

        scheduler_name = self.hparams.lr_scheduler.lower()
        if scheduler_name == "none":
            return optimizer
        if scheduler_name == "step":
            scheduler = schedulers.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_steps,
                gamma=self.hparams.lr_decay_rate,
            )
        elif scheduler_name == "cosine":
            scheduler = schedulers.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.lr_decay_steps,
                eta_min=self.hparams.lr_decay_min_lr,
            )
        else:
            raise ValueError(f"未支持的学习率调度器: {scheduler_name}")
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
