import json
import os, pathlib as pl

from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'


# This code is modified from https://github.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision
class INDataset(Dataset):
    def __init__(self, meta_path, transform):
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __getitem__(self, i):
        image_path = os.path.join(os.pardir, self.data['image_names'][i])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.data['image_labels'][i]
        return image, label

    def __len__(self):
        return len(self.data['image_names'])


def get_mnist_loader(batch_size, num_workers=2, val_split=None, resize=False):
    # Load datasets

    train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True,
                                           download=True)

    mean = train_set.data.float().mean() / 255
    std = train_set.data.float().std() / 255

    # print("Train dataset mean: %.3f, std: %.3f" % (mean, std))

    ts = [transforms.ToTensor(),
          transforms.Normalize(mean, std)]
    if resize:
        ts.append(transforms.Resize(32))

    transform = transforms.Compose(ts)
    train_set.transform = transform

    valid_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True,
                                           transform=transform)

    test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False,
                                          download=True, transform=transform)

    # print((len(train_set.data) + len(test_set.data)))

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    if val_split is not None:

        train_length = len(train_set.data)

        num = int(val_split * train_length)
        train_indices = torch.arange(0, train_length - num)
        valid_indices = torch.arange(train_length - num, train_length)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        return train_loader, test_loader

    return train_loader, valid_loader, test_loader


def get_cifar100_loader(batch_size, num_workers=2, val_split=None):
    # Load datasets
    train_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                              download=True)

    # print(train_set.data.shape)

    mean = train_set.data.mean(axis=(0,1,2)) / 255
    std = train_set.data.std(axis=(0,1,2)) / 255

    print("Train dataset mean: %s, std: %s" % (mean, std))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    train_set.transform = transform

    valid_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                              download=True, transform=transform)

    test_set = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                             download=True, transform=transform)

    test_loader = DataLoader(train_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    if val_split is not None:

        train_length = len(train_set.data)

        num = int(val_split * train_length)
        train_indices = torch.arange(0, train_length - num)
        valid_indices = torch.arange(train_length - num, train_length)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        return train_loader, test_loader

    return train_loader, valid_loader, test_loader


# def get_imagenet_loader(data_path,transform):

def get_imagenet_loader(root, batch_size, num_workers=2):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    # transforms.RandomCrop((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    train_set = INDataset(os.path.abspath(root + '/train.json'), transform)
    valid_set = INDataset(os.path.abspath(root + '/val.json'), transform)
    test_set = INDataset(os.path.abspath(root + '/test.json'), transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, valid_loader
