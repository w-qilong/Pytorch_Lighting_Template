"""可直接运行的示例数据集。

生产模板不提交真实数据文件。这里使用 torchvision FakeData 生成小型假数据，
用于验证训练、验证、测试链路是否正常。
"""

from torchvision import datasets, transforms


class ExampleData(datasets.FakeData):
    def __init__(
        self,
        num_samples: int = 128,
        image_size: int = 28,
        in_channels: int = 1,
        num_classes: int = 10,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
            ]
        )
        super().__init__(
            size=num_samples,
            image_size=(in_channels, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
        )
