import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compression.pruning import get_pruned


class LinearWithAdjustableDropout(nn.Linear):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super().__init__(in_features, out_features)
        self.dropout_rate = dropout_rate

    def forward(self, input: Tensor) -> Tensor:
        out = F.dropout(input, self.dropout_rate, self.training)
        out = super().forward(input)
        return out

    def adjust_dropout_rate(self, name="weight"):
        c_ir, c_i0 = get_pruned(self, name)
        self.dropout_rate = self.dropout_rate * math.sqrt(c_ir / c_i0)


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            LinearWithAdjustableDropout(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            LinearWithAdjustableDropout(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        # probs = F.softmax(logits, dim=1)
        return logits
