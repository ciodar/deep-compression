import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet300(nn.Module):
    def __init__(self, num_classes, activation=F.relu, dropout_rate=0):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 300)
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
    def __init__(self, num_classes, activation=F.relu,dropout_rate=0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        # activation function for hidden layers
        self.activation = activation

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.activation(self.fc(out))
        out = self.dropout(out)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out