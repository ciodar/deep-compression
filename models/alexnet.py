import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        #features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        #classifier
        self.dropout = nn.Dropout(0.5)

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        #conv1
        x = self.activation(self.conv1(x))
        x = self.maxpool(x)
        #conv2
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)
        #conv3 (no maxpool)
        x = self.activation(self.conv3(x))
        #conv4 (no maxpool)
        x = self.activation(self.conv4(x))
        #conv5
        x = self.activation(self.conv5(x))
        x = self.maxpool(x)
        #avgpool
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # classifier
        x = self.dropout(x)
        x = self.activation(self.fc6(x))
        x = self.dropout(x)
        x = self.activation(self.fc7(x))
        logits = self.fc8(x)
        probs = F.softmax(logits, dim=1)
        return logits
