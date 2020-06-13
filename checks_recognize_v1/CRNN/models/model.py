import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter 
import torchvision
from .resnet import * 
from .densenet import *
from .sequence_modeling import BidirectionalLSTM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNBackbones():
    def __init__(self):
        pass 
    def get(self, model_name='vgg16', pretrain=False):
        #print('using cnn backbone: ', model_name)
        
        if model_name == 'resnet':
            model = ResNet_FeatureExtractor(input_channel=1) 
            if pretrain==True:
                print("loading prtrained model............................")
        elif model_name == 'densenet':
            features = list(densenet_cifar().features.children())
            features.append(nn.ReLU(inplace=True))
            model = nn.Sequential(*features)

        return model 
    


class CRNN(nn.Module):

    def __init__(self, class_num, hidden_size=256, backbone='vgg16', pretrain=False):
        super(CRNN, self).__init__()
        self.rnn =  nn.Sequential()
        if backbone == 'resnet':
            input_size=512
        elif backbone=='vgg16':
            input_size=512
        elif backbone == 'densenet':
            input_size=174

        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))
        self.cnn = CNNBackbones().get(backbone, pretrain=pretrain)
        self.rnn = nn.Sequential( 
                    BidirectionalLSTM(input_size, hidden_size, hidden_size),
                    BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
                )

        self.ctc = nn.Linear(hidden_size, class_num)
    def forward(self, x):
        x = self.cnn(x)
        x = self.avgpool(x.permute(0, 3, 1, 2))
        x = x.squeeze(3)
        x = self.rnn(x)
        x = self.ctc(x).log_softmax(2)
        x = x.permute(1, 0, 2)
        return x
