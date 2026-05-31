"""可直接运行的示例数据集。

生产模板不提交真实数据文件。这里使用 torchvision.datasets.FakeData 生成小型假数据，
用于验证训练、验证、测试链路是否正常。

替换为真实数据集时，可以复制本文件并改名，例如 my_dataset.py，然后：
1. 把类名改成 MyDataset；
2. 在 __init__ 中读取 data_dir、split、transform 等参数；
3. 实现或继承 Dataset 所需的 __len__、__getitem__；
4. 在命令行中使用 --train_dataset my_dataset。
"""

from torchvision import datasets, transforms


class ExampleData(datasets.FakeData):
    """模板默认数据集。

    FakeData 会按指定尺寸生成随机图像和随机标签。它不用于真实训练，只用于让用户在
    刚克隆项目时无需下载数据就能检查完整训练流程。
    """

    def __init__(
        self,
        num_samples: int = 128,
        image_size: int = 28,
        in_channels: int = 1,
        num_classes: int = 10,
    ) -> None:
        # transforms.ToTensor 会把 PIL 图像转换为 [0, 1] 的 Tensor。
        # Normalize 使用与通道数匹配的 mean/std，使输出大致落在 [-1, 1]。
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
            ]
        )
        # FakeData 的 image_size 格式是 (C, H, W)，与卷积网络输入一致。
        super().__init__(
            size=num_samples,
            image_size=(in_channels, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
        )
