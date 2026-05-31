# PyTorch Lightning 生产模板

这是一个干净、可扩展的 PyTorch Lightning 项目模板。模板默认使用
`torchvision.datasets.FakeData`，因此克隆后无需下载真实数据即可验证训练链路。

## 设计目标

- 使用当前推荐的 `lightning` 包，而不是旧版 `pytorch_lightning` 导入。
- 让 `main.py` 只负责命令行参数、callbacks 和 `Trainer`。
- 用 `data/DInterface` 统一管理训练、验证、测试 DataLoader。
- 用 `model/MInterface` 统一管理模型、损失函数、指标、优化器和学习率调度器。
- 项目不提交数据、日志、checkpoint、IDE 配置或缓存文件。

## 项目结构

```text
.
├─ main.py
├─ pyproject.toml
├─ data/
│  ├─ __init__.py
│  ├─ data_interface.py
│  └─ example_data.py
├─ model/
│  ├─ __init__.py
│  ├─ model_interface.py
│  └─ example_net.py
└─ tests/
   ├─ test_data_interface.py
   ├─ test_main.py
   └─ test_model_interface.py
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe main.py --fast_dev_run true --accelerator cpu --devices 1 --num_workers 0
```

`fast_dev_run` 会只跑很少的 batch，用于确认数据、模型、loss、日志和 checkpoint
配置没有明显错误。

## 添加新模型

在 `model/` 下新增文件，例如 `resnet_classifier.py`，并定义同名 CamelCase 类：

```python
from torch import nn


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.net(x)
```

然后运行：

```powershell
python main.py --model_name resnet_classifier
```

`MInterface` 会自动把命令行参数中与模型构造函数同名的参数传给模型。

## 添加新数据集

在 `data/` 下新增文件，例如 `my_dataset.py`，并定义 `MyDataset`：

```python
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def __len__(self):
        return 100

    def __getitem__(self, index):
        raise NotImplementedError
```

然后运行：

```powershell
python main.py --train_dataset my_dataset --val_datasets my_dataset --test_datasets my_dataset --data_dir path\to\data
```

## 常用参数

- `--accelerator auto|cpu|gpu`
- `--devices auto|1|0`
- `--precision 32-true|16-mixed|bf16-mixed`
- `--optimizer sgd|adam|adamw`
- `--lr_scheduler none|step|cosine`
- `--metric accuracy|recall`
- `--val_datasets` 和 `--test_datasets` 支持传入多个数据集名称。

## 开发约定

- 数据集文件使用 `snake_case.py`，类名使用 `CamelCase`。
- 模型文件使用 `snake_case.py`，类名使用 `CamelCase`。
- 不要提交真实数据、日志、checkpoint、缓存或 IDE 配置。
- 扩展点附近保留中文注释，业务代码应优先保持简洁。
