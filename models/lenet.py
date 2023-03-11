import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet300(nn.Module):
    def __init__(self, num_classes, activation=F.relu, dropout_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        # activation function for hidden layers
        self.activation = activation

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.activation(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False, activation=F.relu, dropout=0):
        super().__init__()
        self.grayscale = grayscale
        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.conv1 = nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5, stride=1, padding=0)
        self.fc = nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels)
        self.fc1 = nn.Linear(120 * in_channels, 84 * in_channels)
        self.fc2 = nn.Linear(84 * in_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        # activation function for hidden layers
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.activation(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.activation(self.fc(out))
        out = self.activation(self.fc1(out))
        out = self.fc2(out)
        return out
