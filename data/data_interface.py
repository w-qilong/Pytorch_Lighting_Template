"""数据模块统一入口。

本文件只负责把用户在命令行中指定的数据集名称，转换为 Lightning 可用的
DataLoader。新增数据集时，通常只需要在 data/ 下添加一个 Dataset 类文件，
不需要重写 LightningDataModule。

使用步骤：
1. 在 data/ 下新增一个 snake_case.py 文件，例如 my_dataset.py；
2. 在文件中定义同名 CamelCase 类，例如 MyDataset；
3. 让该类继承 torch.utils.data.Dataset，并实现 __len__ 与 __getitem__；
4. 在命令行中指定 --train_dataset my_dataset。
"""

import importlib
import inspect
from collections.abc import Sequence

import lightning as L
from torch.utils.data import DataLoader


def _as_list(value: str | Sequence[str]) -> list[str]:
    """把命令行传入的单个名称或多个名称统一成 list。

    argparse 在 nargs="+" 时会返回 list；但默认值或手动调用 DInterface 时
    可能传入单个字符串。统一成 list 后，后续 val/test dataloader 逻辑更简单。
    """
    if isinstance(value, str):
        return [value]
    return list(value)


class DInterface(L.LightningDataModule):
    """模板级 DataModule。

    约定：数据集文件使用 snake_case 命名，类名使用 CamelCase。
    例如 data/example_data.py 中应定义 ExampleData。

    这个类把“数据集怎么实例化”和“DataLoader 怎么创建”集中起来。用户新增数据集时
    只需要关注 Dataset 本身，不需要在每个数据集里重复写 train_dataloader、
    val_dataloader、test_dataloader。
    """

    def __init__(
        self,
        train_dataset: str = "example_data",
        val_datasets: str | Sequence[str] = ("example_data",),
        test_datasets: str | Sequence[str] = ("example_data",),
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        # 训练集通常只有一个；验证集和测试集可以有多个，用于跨数据集评估。
        self.train_dataset_name = train_dataset
        self.val_dataset_names = _as_list(val_datasets)
        self.test_dataset_names = _as_list(test_datasets)
        # batch_size 和 num_workers 是所有 DataLoader 共用的基础配置。
        self.batch_size = batch_size
        self.num_workers = num_workers
        # kwargs 保存 main.py 中传入的其它参数。真正创建 Dataset 时，会按构造函数
        # 签名自动筛选，只有 Dataset 声明过的参数才会被传入。
        self.kwargs = kwargs

    def _load_dataset_class(self, dataset_name: str):
        """根据数据集文件名动态导入 Dataset 类。

        命名转换规则：
        - example_data -> ExampleData
        - my_custom_dataset -> MyCustomDataset

        因此新增数据集时，文件名和类名必须匹配这个规则。
        """
        class_name = "".join(part.capitalize() for part in dataset_name.split("_"))
        try:
            module = importlib.import_module(f".{dataset_name}", package=__package__)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"无法加载数据集 data/{dataset_name}.py 中的 {class_name} 类。"
            ) from exc

    def _instantiate_dataset(self, dataset_name: str):
        """实例化指定数据集。

        main.py 会把所有命令行参数都传进 DInterface，其中很多参数只属于模型或
        Trainer。这里通过 inspect.signature 读取 Dataset.__init__ 的参数列表，
        只传入 Dataset 真正需要的参数，避免无关 CLI 参数污染。
        """
        dataset_cls = self._load_dataset_class(dataset_name)
        signature = inspect.signature(dataset_cls.__init__)
        accepted = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        dataset_kwargs = {name: self.kwargs[name] for name in accepted if name in self.kwargs}
        return dataset_cls(**dataset_kwargs)

    def setup(self, stage: str | None = None) -> None:
        """按 Lightning 的阶段创建 Dataset 实例。

        Lightning 会在 fit/test/predict 前自动调用 setup。这里按 stage 懒加载数据集：
        - fit 阶段创建训练集和验证集；
        - test/predict 阶段创建测试集；
        - stage=None 时一次性创建全部数据集，便于手动调试或测试。
        """
        if stage in (None, "fit"):
            self.train_set = self._instantiate_dataset(self.train_dataset_name)
            self.val_sets = [self._instantiate_dataset(name) for name in self.val_dataset_names]

        if stage in (None, "test", "predict"):
            self.test_sets = [self._instantiate_dataset(name) for name in self.test_dataset_names]

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        """为任意 Dataset 创建 DataLoader。

        训练集需要 shuffle=True；验证、测试、预测通常使用 shuffle=False，保证评估结果
        可复现并便于定位样本。
        """
        # persistent_workers 只能在 num_workers > 0 时启用，否则 PyTorch 会报错。
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """返回训练 DataLoader。

        Lightning 在 trainer.fit 中自动调用本方法，并把 batch 传给
        model.training_step。
        """
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        """返回一个或多个验证 DataLoader。

        即使只有一个验证集，也返回 list，便于未来无缝扩展多验证集场景。
        Lightning 记录多 dataloader 指标时会追加 /dataloader_idx_N。
        """
        return [self._loader(dataset, shuffle=False) for dataset in self.val_sets]

    def test_dataloader(self) -> list[DataLoader]:
        """返回一个或多个测试 DataLoader，供 trainer.test 使用。"""
        return [self._loader(dataset, shuffle=False) for dataset in self.test_sets]

    def predict_dataloader(self) -> list[DataLoader]:
        """预测阶段复用测试集配置。

        如果项目需要单独的预测集，可以参考 test_dataloader 新增 predict_dataset 参数。
        """
        return self.test_dataloader()
