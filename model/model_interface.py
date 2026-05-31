"""模型模块统一入口。

MInterface 封装 LightningModule 的通用训练逻辑。新增模型时，只需要在
model/ 下添加普通 nn.Module，再通过 --model_name 指定即可。

使用步骤：
1. 在 model/ 下新增一个 snake_case.py 文件，例如 resnet_classifier.py；
2. 在文件中定义同名 CamelCase 类，例如 ResnetClassifier；
3. 让该类继承 torch.nn.Module 并实现 forward；
4. 在 main.py 的 argparse 中补充模型构造函数需要的超参数；
5. 在命令行中指定 --model_name resnet_classifier。
"""

import importlib
import inspect

import lightning as L
import torch
from torch import nn
from torch.optim import lr_scheduler as schedulers
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall


class MInterface(L.LightningModule):
    """通用 LightningModule，负责模型、损失、指标和优化器配置。

    用户通常不需要为每个模型重新写 LightningModule。只要新模型是普通的
    nn.Module，本类就能统一提供训练、验证、测试、预测、优化器和日志逻辑。
    """

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
        # 保存所有传入超参数。Lightning 会把它们写入 checkpoint，便于复现实验。
        self.save_hyperparameters()
        # 动态加载真正的业务模型，例如 ExampleNet、ResnetClassifier 等。
        self.model = self._load_model(model_name, kwargs)
        # 根据字符串参数创建损失函数和指标对象，便于命令行切换。
        self.loss_function = self._build_loss(loss)
        self.metric_name = metric
        # 验证集和测试集使用独立 metric 实例，避免状态互相污染。
        self.valid_metric = self._build_metric(metric, num_classes)
        self.test_metric = self._build_metric(metric, num_classes)

    def _load_model(self, model_name: str, extra_kwargs: dict):
        """根据模型文件名动态导入 nn.Module 类并实例化。

        命名转换规则：
        - example_net -> ExampleNet
        - resnet_classifier -> ResnetClassifier

        本函数还会自动读取模型构造函数签名，只传入模型真正声明过的参数。
        这样 main.py 可以统一维护参数列表，而不会因为多余参数导致模型初始化失败。
        """
        class_name = "".join(part.capitalize() for part in model_name.split("_"))
        try:
            module = importlib.import_module(f".{model_name}", package=__package__)
            model_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"无法加载模型 model/{model_name}.py 中的 {class_name} 类。") from exc

        # inspect.signature 会读取 __init__ 参数，例如 in_channels、num_classes、hidden_dim。
        signature = inspect.signature(model_cls.__init__)
        accepted = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        # 先从 extra_kwargs 中拿参数。extra_kwargs 是 main.py 传入的命令行参数集合。
        model_kwargs = {name: extra_kwargs[name] for name in accepted if name in extra_kwargs}
        # 再从 self.hparams 中补参数，保留 Lightning 保存的超参数访问方式。
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
        """根据命令行中的 --loss 创建损失函数。

        扩展方法：如果需要新增 focal_loss、dice_loss 等，在这里增加分支即可。
        对于需要额外超参数的复杂 loss，可以改成普通实例方法并从 self.hparams 读取。
        """
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
        """根据命令行中的 --metric 创建 torchmetrics 指标。

        torchmetrics 会在 Lightning 中自动处理设备迁移、跨 batch 状态累积和 epoch
        结束时的统计。新增指标时，请优先选择 torchmetrics 中已有实现。
        """
        metric = metric.lower()
        if metric == "accuracy":
            return MulticlassAccuracy(num_classes=num_classes)
        if metric == "recall":
            return MulticlassRecall(num_classes=num_classes)
        raise ValueError(f"未支持的评价指标: {metric}")

    def forward(self, images):
        """前向传播入口。

        Lightning 会在 training_step、validation_step、test_step 中通过 self(images)
        调用本方法。这里不写 loss 或 metric，只返回模型预测结果。
        """
        return self.model(images)

    def _shared_eval_step(self, batch, stage: str):
        """验证和测试共用的单 batch 逻辑。

        stage 取 "val" 或 "test"，用于生成不同日志名称：
        - val_loss / val_accuracy
        - test_loss / test_accuracy
        """
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        metric = self.valid_metric if stage == "val" else self.test_metric
        # torchmetrics 的 update 会累积当前 batch 的状态，compute 由 Lightning 日志系统触发。
        metric.update(logits, labels)
        # batch_size 用于 Lightning 在 epoch 级别正确加权聚合 loss/metric。
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=images.size(0))
        self.log(f"{stage}_{self.metric_name}", metric, prog_bar=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        """训练阶段的单 batch 逻辑。

        返回值必须是 loss Tensor，Lightning 会自动执行 backward、optimizer.step、
        zero_grad 等优化流程。除非需要手动优化，一般不要在这里直接调用 optimizer。
        """
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证阶段的单 batch 逻辑。

        dataloader_idx 用于兼容多个验证集；当前通用逻辑不需要显式使用它。
        """
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """测试阶段的单 batch 逻辑，通常与验证逻辑一致。"""
        return self._shared_eval_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """预测阶段的单 batch 逻辑。

        默认返回 softmax 后的类别概率。若你的任务是回归、分割或检测，请根据输出格式
        修改这里。
        """
        images, _ = batch
        return self(images).softmax(dim=1)

    def configure_optimizers(self):
        """创建优化器和可选学习率调度器。

        Lightning 会自动调用本方法，并根据返回值管理 optimizer/scheduler。
        扩展方法：
        - 新增优化器：在 optimizer_name 分支中创建对应 torch.optim 对象；
        - 新增调度器：在 scheduler_name 分支中创建对应 lr_scheduler 对象。
        """
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
        # 不使用学习率调度器时，Lightning 允许只返回 optimizer。
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
        # 返回字典可以让 Lightning 清楚识别 optimizer 与 lr_scheduler 的关系。
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
