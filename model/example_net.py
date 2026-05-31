"""最小 CNN 示例模型。

该模型用于证明模板可运行。实际项目中可复制本文件并改名，例如
resnet_classifier.py 中定义 ResnetClassifier。

新增模型时请注意：
1. 文件名使用 snake_case，例如 my_model.py；
2. 类名使用 CamelCase，例如 MyModel；
3. __init__ 中需要的参数应同步添加到 main.py 的 argparse；
4. forward 只负责输入到输出，不要在普通 nn.Module 中写训练循环。
"""

from torch import nn


class ExampleNet(nn.Module):
    """面向 28x28 单通道图像的简单分类网络。

    输入形状默认是 [batch, 1, 28, 28]，输出形状是 [batch, num_classes]。
    该结构刻意保持简单，方便用户快速看懂模板如何连接数据、模型和 Lightning。
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        # 特征提取部分：两层卷积 + ReLU + 池化。
        # 每次 MaxPool2d(kernel_size=2) 会把宽高减半：28 -> 14 -> 7。
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # 分类头：把 32 个 7x7 特征图展平，再映射到类别数。
        # 如果修改 image_size 或池化次数，请同步调整 Linear 的输入维度。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """执行模型前向传播。

        参数 x 是一批图像 Tensor；返回值是未经过 softmax 的 logits。
        CrossEntropyLoss 期望输入 logits，因此这里不要手动加 softmax。
        """
        x = self.features(x)
        return self.classifier(x)
