import json
import os

from PIL import Image
from torch.utils.data import Dataset


# def get_mnist_loader():


# def get_cifar100_loader():
#
#
#
# def get_imagenet_loader(data_path,transform):


# This code is modified from https://github.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision
class INDataset(Dataset):
    def __init__(self, data_path, transform):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __getitem__(self, i):
        image_path = os.path.join(self.data['image_names'][i])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.data['image_labels'][i]
        return image, label

    def __len__(self):
        return len(self.data['image_names'])
