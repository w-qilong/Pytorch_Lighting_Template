"""最小 CNN 示例模型。

该模型用于证明模板可运行。实际项目中可复制本文件并改名，例如
`resnet_classifier.py` 中定义 `ResnetClassifier`。
"""

from torch import nn


class ExampleNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
