"""数据模块统一入口。

本文件只负责把用户在命令行中指定的数据集名称，转换为 Lightning 可用的
DataLoader。新增数据集时，通常只需要在 data/ 下添加一个 Dataset 类文件。
"""

import importlib
import inspect
from collections.abc import Sequence

import lightning as L
from torch.utils.data import DataLoader


def _as_list(value: str | Sequence[str]) -> list[str]:
    """把命令行传入的单个名称或多个名称统一成 list。"""
    if isinstance(value, str):
        return [value]
    return list(value)


class DInterface(L.LightningDataModule):
    """模板级 DataModule。

    约定：数据集文件使用 snake_case 命名，类名使用 CamelCase。
    例如 data/example_data.py 中应定义 ExampleData。
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
        self.train_dataset_name = train_dataset
        self.val_dataset_names = _as_list(val_datasets)
        self.test_dataset_names = _as_list(test_datasets)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def _load_dataset_class(self, dataset_name: str):
        class_name = "".join(part.capitalize() for part in dataset_name.split("_"))
        try:
            module = importlib.import_module(f".{dataset_name}", package=__package__)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"无法加载数据集 data/{dataset_name}.py 中的 {class_name} 类。"
            ) from exc

    def _instantiate_dataset(self, dataset_name: str):
        """只把 Dataset 构造函数真正声明过的参数传进去，避免无关 CLI 参数污染。"""
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
        if stage in (None, "fit"):
            self.train_set = self._instantiate_dataset(self.train_dataset_name)
            self.val_sets = [self._instantiate_dataset(name) for name in self.val_dataset_names]

        if stage in (None, "test", "predict"):
            self.test_sets = [self._instantiate_dataset(name) for name in self.test_dataset_names]

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        # persistent_workers 只能在 num_workers > 0 时启用。
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        return [self._loader(dataset, shuffle=False) for dataset in self.val_sets]

    def test_dataloader(self) -> list[DataLoader]:
        return [self._loader(dataset, shuffle=False) for dataset in self.test_sets]

    def predict_dataloader(self) -> list[DataLoader]:
        return self.test_dataloader()
