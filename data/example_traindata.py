# todo: Here, we use minist for example

from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np


def load_data(data_folder, data_name, label_name):
    '''
    unzip minist dataset
    :param data_folder: the root folder of MNIST dataset
    :param data_name: the file of image data
    :param label_name: label_name：the label of image
    :return:
    '''

    with open(os.path.join(data_folder, label_name), 'rb') as label_path:  # rb means read binary data.
        train_label = np.frombuffer(label_path.read(), np.uint8, offset=8)  # The first 8 bytes are not data content

    with open(os.path.join(data_folder, data_name), 'rb') as img_path:
        train_data = np.frombuffer(
            img_path.read(), np.uint8, offset=16).reshape(len(train_label), 28,
                                                          28)  # The first 16 bytes are not data content
    return train_data, train_label


# define transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,), (0.3081,))
])


class ExampleTraindata(Dataset):
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.dirname(__file__))
        self.folder = os.path.join(self.dir_path, 'example_Minist_data/MNIST/raw')
        self.data_file = 'train-images-idx3-ubyte'
        self.label_file = 'train-labels-idx1-ubyte'
        self.train_data, self.train_labels = load_data(self.folder, self.data_file, self.label_file)
        self.transforms = transforms

    def __getitem__(self, index):
        img, target = self.train_data[index], int(self.train_labels[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.train_data)

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     minist_train_dataset=ExampleTraindata()
#     print(len(minist_train_dataset))
#
#     dataloader = DataLoader(dataset=minist_train_dataset, batch_size=2, shuffle=False, num_workers=4)
#     for batch_idx, batch in enumerate(dataloader):
#         image, label = batch
#         print(image.shape)
#         print(label.shape)
#         print(label)
#         break
