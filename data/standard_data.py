# todo: in this file, define your dataset.

import torch.utils.data as data
from torchvision import transforms as T

# you can define transforms you want to use for images here. The mean and std of ImageNet-1k are here.
IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]}

# you can add more transform to transforms
train_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    # Number of augmentation transformations to apply sequentially
    # Paper:  <https://arxiv.org/abs/1909.13719>
    T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),
])

valid_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std'])])


# define your standard dataset
class StandardData(data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
