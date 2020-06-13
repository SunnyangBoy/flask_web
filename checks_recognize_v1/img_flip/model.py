import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        # 224x144
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # 112x72
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        # 56x36
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)

        # 28x18
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)

        # 14x9
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        # 7x4
        self.fc1 = nn.Linear(512 * 14 * 9, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        # x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, 2, 2)

        #  x = F.relu(self.conv11(x))
        #  x = F.relu(self.conv12(x))
        #  x = F.relu(self.conv13(x))
        # x = F.max_pool2d(x, 2, 2)

        return x


class DetectAngleModel(torch.nn.Module):
    def __init__(self):
        super(DetectAngleModel, self).__init__()
        self.cnn = torchvision.models.vgg16(pretrained=False).features
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)
        # self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # #112x72
        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        # #56x36
        # self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        # self.conv7 = nn.Conv2d(256, 256, 3, padding=1)

        # #28x18
        # self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        # self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv10 = nn.Conv2d(512, 512, 3, padding=1)

        # #14x9
        # self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        # #7x4
        # self.fc1 = nn.Linear(512*14*14, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 2)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        # x = self.cnn(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        #   print(x.size())
        x = x.view(batch_size, -1)
        # print(x.size())
        # print(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x
