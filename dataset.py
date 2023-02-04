import json
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'


def get_mnist_loader(batch_size, num_workers=2, resize=False):
    # Load datasets

    trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True,
                                          download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    mean = trainset.data.float().mean() / 255
    std = trainset.data.float().std() / 255

    # print("Train dataset mean: %.3f, std: %.3f" % (mean, std))

    ts = [transforms.ToTensor(),
          transforms.Normalize(mean, std)]
    if resize:
        ts.append(transforms.Resize(32))

    transform = transforms.Compose(ts)
    trainset.transform = transform

    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def get_cifar100_loader(batch_size, num_workers=2):
    # Load datasets
    trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                             download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    mean = trainset.data.float().mean() / 255
    std = trainset.data.float().std() / 255

    # print("Train dataset mean: %.3f, std: %.3f" % (mean, std))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    trainset.transform = transform

    testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader,testloader


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
        # image = self.transform(image)
        label = self.data['image_labels'][i]
        return image, label

    def __len__(self):
        return len(self.data['image_names'])
