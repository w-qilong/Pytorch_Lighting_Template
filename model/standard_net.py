# todo: In this file, you can define your multiple model.

import torch
from torch import nn
# import timm # If we need to use pretrained model, we can use timm library to quickly create a model.


class StandardNet(nn.Module):
    """ If you want to use pretrained model, or simply the standard structure implemented
        by Pytorch official, please use this template. It enables you to easily control whether
        use or not the pretrained weights, and whether to freeze the internal layers or not,
        and the in/out channel numbers, resnet version. This is made for resnet, but you can
        also adapt it to other structures by changing the `torch.hub.load` content.
    """

    def __init__(self, in_channel=1, out_channel=10, freeze=False, pretrained=False, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        pass
