import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compression.pruning import get_pruned

class LinearWithAdjustableDropout(nn.Linear):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super().__init__(in_features,out_features)
        self.dropout_rate = dropout_rate

    def forward(self, input: Tensor) -> Tensor:
        out = super().forward(input)
        out = F.dropout(out,self.dropout_rate,self.training)
        return out

    def adjust_dropout_rate(self,name="weight"):
        c_ir, c_i0 = get_pruned(self,name)
        self.dropout_rate = self.dropout_rate * math.sqrt(c_ir/c_i0)

class AlexNet(nn.Module):

    def __init__(self, num_classes, dropout_rate=0.5):
        super(AlexNet, self).__init__()
        # features
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.LRN = nn.LocalResponseNorm(5, 0.0001, 0.75)

        # classifier
        self.dropout_rate = dropout_rate
        self.dropout = F.dropout

        self.fc6 = LinearWithAdjustableDropout(256 * 6 * 6, 4096)
        self.fc7 = LinearWithAdjustableDropout(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # conv1
        x = self.activation(self.conv1(x))
        x = self.LRN(x)
        x = self.maxpool(x)
        # conv2
        x = self.activation(self.conv2(x))
        x = self.LRN(x)
        x = self.maxpool(x)
        # conv3 (no maxpool)
        x = self.activation(self.conv3(x))
        # conv4 (no maxpool)
        x = self.activation(self.conv4(x))
        # conv5
        x = self.activation(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # classifier
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        logits = self.fc8(x)
        probs = F.softmax(logits, dim=1)
        return logits

