import torch
from torch import nn


# Here, define a simple model for minist dataset for example
class ExampleNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = torch.nn.Sequential(
            # The size of the input picture is 28x28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the feature map is 14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the feature map is 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        output = self.model(x)
        return output
