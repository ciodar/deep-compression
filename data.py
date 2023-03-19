import json
import os, pathlib as pl

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class MnistDataLoader(BaseDataLoader):
    def __init__(self, batch_size, data_dir=DATA_DIR, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 resize=False):
        # calculated dataset mean and variance for standardization
        # train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True,
        #                                        download=True)
        #
        # mean = train_set.data.float().mean() / 255 #0.1307
        # std = train_set.data.float().std() / 255 #0.3081

        self.resize = resize
        ts = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        if resize:
            ts.append(transforms.Resize(32, antialias=True))
        transform = transforms.Compose(ts)
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(root=self.data_dir, train=training, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar100DataLoader(BaseDataLoader):
    # calculated dataset mean and variance for standardization
    def __init__(self, batch_size, data_dir=DATA_DIR, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        train_set = datasets.CIFAR100(root=DATA_DIR, train=True,
                                      download=True)

        mean = train_set.data.mean() / 255
        std = train_set.data.std() / 255

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(root=self.data_dir, train=training, download=True, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# This code is modified from https://github.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision
class MINDataset(Dataset):
    def __init__(self, root, train, transform):
        meta_dir = pl.Path(root).joinpath('mini-in')
        if train:
            file = meta_dir / 'train.json'
        else:
            file = meta_dir / 'test.json'
        with open(file) as f:
            self.data = json.load(f)
        self.transform = transform

    def __getitem__(self, i):
        image_path = self.data['image_names'][i]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.data['image_labels'][i]
        return image, label

    def __len__(self):
        return len(self.data['image_names'])


class ImagenetteDataLoader(BaseDataLoader):
    def __init__(self, batch_size, data_dir=os.path.join(DATA_DIR, 'imagenette2'), shuffle=True, validation_split=0.0,
                 num_workers=1,
                 training=True):
        traindir = os.path.join(data_dir, 'train')
        testdir = os.path.join(data_dir, 'val')

        print(os.path.abspath(traindir))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if training:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        self.dataset = datasets.ImageFolder(
            traindir if training else testdir,
            transform
        )

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ImagenetDataLoader(BaseDataLoader):
    # calculated dataset mean and variance for standardization
    def __init__(self, batch_size, data_dir=DATA_DIR, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        # transforms.RandomCrop((64, 64)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        self.data_dir = data_dir
        self.dataset = MINDataset(root=self.data_dir, train=training, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
