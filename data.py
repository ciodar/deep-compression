import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset, Dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


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


class ImagenetDataLoader(BaseDataLoader):
    def __init__(self, batch_size, data_dir, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True):
        if training:
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Todo: define augmentation techniques, different for train and validation.
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

        dataset = datasets.ImageFolder(
            data_dir,
            transform=None
        )
        self.n_samples = len(dataset)
        self.train_dataset, self.valid_dataset = self._split_dataset(dataset, validation_split)
        validation_split = 0.0

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_dataset(self, dataset, split):
        if split == 0.0:
            return MyDataset(dataset, self.transform['val']), None

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

        train_dataset = MyDataset(Subset(dataset, train_idx), self.transform['train'])
        valid_dataset = MyDataset(Subset(dataset, valid_idx), self.transform['val'])

        self.n_samples = len(train_idx)

        return train_dataset, valid_dataset

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, batch_size=self.init_kwargs['batch_size'],
                              shuffle=False, num_workers=self.init_kwargs['num_workers'])
