import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm
import time
import argparse
import ast
from dataset import DeepFashion
from torch.nn import DataParallel
#from sync_batchnorm import DataParallelWithCallback as DataParallel
import numpy as np
#from resnet import *
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import numpy.matlib

class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.select = {
            '1': 'conv1_1',  # [batch_size, 64, 224, 224]
            '3': 'conv1_2',  # [batch_size, 64, 224, 224]
            '4': 'pooled_1',  # [batch_size, 64, 112, 112]
            '6': 'conv2_1',  # [batch_size, 128, 112, 112]
            '8': 'conv2_2',  # [batch_size, 128, 112, 112]
            '9': 'pooled_2',  # [batch_size, 128, 56, 56]
            '11': 'conv3_1',  # [batch_size, 256, 56, 56]
            '13': 'conv3_2',  # [batch_size, 256, 56, 56]
            '15': 'conv3_3',  # [batch_size, 256, 56, 56]
            '16': 'pooled_3',  # [batch_size, 256, 28, 28]
            '18': 'conv4_1',  # [batch_size, 512, 28, 28]
            '20': 'conv4_2',  # [batch_size, 512, 28, 28]
            '22': 'conv4_3',  # [batch_size, 512, 28, 28]
            '23': 'pooled_4',  # [batch_size, 512, 14, 14]
            '25': 'conv5_1',  # [batch_size, 512, 14, 14]
            '27': 'conv5_2',  # [batch_size, 512, 14, 14]
            '29': 'conv5_3',  # [batch_size, 512, 14, 14]
            '30': 'pooled_5',  # [batch_size , 512, 7, 7]
        }
        self.vgg = models.vgg16(pretrained=True).features

    def forward(self, x):
        ret = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                ret[self.select[name]] = x
#                 print(self.select[name], x.size())
#                 print(name, layer)
        return ret

class landmrak_vgg16(nn.Module):
    def __init__(self):
        super(landmrak_vgg16, self).__init__()
        self.vgg16_extractor = VGG16Extractor()
        self.conv1 = nn.Conv2d(512, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.conv9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv10 = nn.Conv2d(16, 8, 1, 1, 0)
    def forward(self, x):
        x=self.vgg16_extractor(x)['conv4_3']
        #print('11111111111111111111111111111111111111111111')
        #print(x.size())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.upconv1(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        #print('2222222222222222222222222222222222222222222222222')
        #print(x.size())
        lm_pos_map = x
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)
        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped.cpu(), dim=2), (pred_h, pred_w))
        #lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        return lm_pos_map, lm_pos_output

class landmrak_vgg16_sigmoid(nn.Module):
    def __init__(self):
        super(landmrak_vgg16_sigmoid, self).__init__()
        self.vgg16_extractor = VGG16Extractor()
        self.conv1 = nn.Conv2d(512, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.conv9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv10 = nn.Conv2d(16, 8, 1, 1, 0)
    def forward(self, x):
        x=self.vgg16_extractor(x)['conv4_3']
        #print('11111111111111111111111111111111111111111111')
        #print(x.size())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.upconv1(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        #print('2222222222222222222222222222222222222222222222222')
        #print(x.size())
        x=nn.Sigmoid(x)
        lm_pos_map = x
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)
        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped.cpu(), dim=2), (pred_h, pred_w))
        #lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        return lm_pos_map, lm_pos_output



class myAlexNet(nn.Module):
    def __init__(self, num_category=50,num_attribute=1000):
        super(myAlexNet, self).__init__()
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),

        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x_category = self.classifier_category(x)
        x_attribute = self.classifier_attribute(x)
        return x_category, x_attribute

class myVGG16(nn.Module):
    def __init__(self, num_category=50,num_attribute=1000, init_weights=True):
        super(myVGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),

        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute),
            nn.Sigmoid()
        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        x_category = self.classifier_category(x)
        x_attribute = self.classifier_attribute(x)
        return x_category, x_attribute
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class myVGG19(nn.Module):
    def __init__(self, num_category=50,num_attribute=1000):
        super(myVGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),

        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        x_category = self.classifier_category(x)
        x_attribute = self.classifier_attribute(x)
        return x_category, x_attribute

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output=torch.zeros((1,8,self.out_features)).cuda()
        for i in range(input.size(0)):
            support = torch.mm(input[i], self.weight)
            output = torch.cat([output,torch.unsqueeze(torch.spmm(adj, support),0)],0)
        output=output[1:]
        if self.bias is not None:
            return output + self.bias
        else:
            return output




    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class model_alexnet_v0(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_alexnet_v0, self).__init__()
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

        self.gc1 = GraphConvolution(256, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute),
            nn.Sigmoid()
        )

    def forward(self, x,landmarks,adj):
        x_mid = self.features(x)

        x1=torch.zeros((1,8,256)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,256)).cuda()
            for j in range(8):
                #print(landmarks[i][j][0],landmarks[i][j][1],landmarks[i][j][2],landmarks[i][j][3])
                #if landmarks[i][j][1]==1:
                if landmarks[i][j][0]==1:
                    xxx = landmarks[i][j][1]
                    xxx = xxx / 224.
                    xxx = int(xxx * x_mid.size(2))
                    yyy = landmarks[i][j][2]
                    yyy = yyy / 224.
                    yyy = int(yyy * x_mid.size(3))
                    # print(i, xxx, yyy)
                    v = x_mid[i, :, xxx, yyy]
                else:
                    v = torch.zeros((256))
                v=v.view(1,1,256).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 256 * 6 * 6)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        return x_category,x_attribute

class model_alexnet_v1(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_alexnet_v1, self).__init__()
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

        self.gc1 = GraphConvolution(256, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute),
            nn.Sigmoid()
        )

    def forward(self, x,landmarks,adj):
        x_mid = self.features(x)

        x1=torch.zeros((1,8,256)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,256)).cuda()
            for j in range(8):
                if landmarks[i][j][1]==1:
                    xxx=landmarks[i][j][2]
                    xxx=xxx/224.
                    xxx=int(xxx*x_mid.size(2))
                    yyy=landmarks[i][j][3]
                    yyy=yyy/224.
                    yyy=int(yyy*x_mid.size(3))
                    #print(i, xxx, yyy)
                    v=x_mid[i,:,xxx,yyy]
                else:
                    v=torch.zeros((256))
                v=v.view(1,1,256).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 256 * 6 * 6)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        return x_category,x_attribute

class model_alexnet_v2(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_alexnet_v2, self).__init__()
        self.num_category=num_category
        self.num_attribute=num_attribute
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

        self.gc1 = GraphConvolution(256, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute*2),
        )

    def forward(self, x,landmarks,adj):
        x_mid = self.features(x)

        x1=torch.zeros((1,8,256)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,256)).cuda()
            for j in range(8):
                if landmarks[i][j][1]==1:
                    xxx=landmarks[i][j][2]
                    xxx=xxx/224.
                    xxx=int(xxx*x_mid.size(2))
                    yyy=landmarks[i][j][3]
                    yyy=yyy/224.
                    yyy=int(yyy*x_mid.size(3))
                    #print(i, xxx, yyy)
                    v=x_mid[i,:,xxx,yyy]
                else:
                    v=torch.zeros((256))
                v=v.view(1,1,256).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 256 * 6 * 6)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        x_attribute=x_attribute.view(x_attribute.size(0),2,self.num_attribute)
        return x_category,x_attribute

class model_vgg16_v1(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_vgg16_v1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.gc1 = GraphConvolution(512, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute),
            nn.Sigmoid()
        )

    def forward(self, x,landmarks,adj):
        x_mid = self.features(x)

        x1=torch.zeros((1,8,512)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,512)).cuda()
            for j in range(8):
                if landmarks[i][j][1]==1:
                    xxx=landmarks[i][j][2]
                    xxx=xxx/224.
                    xxx=int(xxx*x_mid.size(2))
                    yyy=landmarks[i][j][3]
                    yyy=yyy/224.
                    yyy=int(yyy*x_mid.size(3))
                    #print(i, xxx, yyy)
                    v=x_mid[i,:,xxx,yyy]
                else:
                    v=torch.zeros((512))
                v=v.view(1,1,512).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 512 * 7 * 7)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        return x_category,x_attribute

class model_vgg16_v2(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_vgg16_v2, self).__init__()
        self.num_category = num_category
        self.num_attribute = num_attribute
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.gc1 = GraphConvolution(512, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute*2),
        )

    def forward(self, x,landmarks,adj):
        x_mid = self.features(x)

        x1=torch.zeros((1,8,512)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,512)).cuda()
            for j in range(8):
                if landmarks[i][j][1]==1:
                    xxx=landmarks[i][j][2]
                    xxx=xxx/224.
                    xxx=int(xxx*x_mid.size(2))
                    yyy=landmarks[i][j][3]
                    yyy=yyy/224.
                    yyy=int(yyy*x_mid.size(3))
                    #print(i, xxx, yyy)
                    v=x_mid[i,:,xxx,yyy]
                else:
                    v=torch.zeros((512))
                v=v.view(1,1,512).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 512 * 7 * 7)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        x_attribute = x_attribute.view(x_attribute.size(0), 2, self.num_attribute)
        return x_category,x_attribute

#using given landmarks
class model_vgg16_all_v1(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_vgg16_all_v1, self).__init__()
        self.num_category = num_category
        self.num_attribute = num_attribute
        self.vgg16_extractor = VGG16Extractor()

        self.up=nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, 1, 0),
        )
        self.gc1 = GraphConvolution(512, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute*2),
        )

    def forward(self, x,landmarks,adj):
        ret={}
        x_lm=self.vgg16_extractor(x)['conv4_3']
        x_lm =self.up(x_lm)
        lm_pos_map = x_lm
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)
        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped.cpu(), dim=2), (pred_h, pred_w))
        # lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        ret['lm_pos_map']=lm_pos_map
        ret['lm_pos_output'] = lm_pos_output


        x_mid =self.vgg16_extractor(x)['pooled_5']
        x1=torch.zeros((1,8,512)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,512)).cuda()
            for j in range(8):
                if landmarks[i][j][1]==1:
                    xxx=landmarks[i][j][2]
                    xxx=xxx/224.
                    xxx=int(xxx*x_mid.size(2))
                    yyy=landmarks[i][j][3]
                    yyy=yyy/224.
                    yyy=int(yyy*x_mid.size(3))
                    #print(i, xxx, yyy)
                    v=x_mid[i,:,xxx,yyy]
                else:
                    v=torch.zeros((512))
                v=v.view(1,1,512).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 512 * 7 * 7)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        x_attribute = x_attribute.view(x_attribute.size(0), 2, self.num_attribute)

        ret['category_output']=x_category
        ret['attr_output']=x_attribute
        return ret

#using predicted landmarks
class model_vgg16_all_v2(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_vgg16_all_v2, self).__init__()
        self.num_category = num_category
        self.num_attribute = num_attribute
        self.vgg16_extractor = VGG16Extractor()

        self.up=nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, 1, 0),
        )
        self.gc1 = GraphConvolution(512, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_attribute*2),
        )

    def forward(self, x,landmarks,adj):
        ret={}
        x_lm=self.vgg16_extractor(x)['conv4_3']
        x_lm =self.up(x_lm)
        lm_pos_map = x_lm
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)
        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped.cpu(), dim=2), (pred_h, pred_w))
        # lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        ret['lm_pos_map']=lm_pos_map
        ret['lm_pos_output'] = lm_pos_output


        x_mid =self.vgg16_extractor(x)['pooled_5']
        x1=torch.zeros((1,8,512)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,512)).cuda()
            for j in range(8):
                xxx = lm_pos_output[i][j][0]
                yyy=lm_pos_output[i][j][1]
                if lm_pos_map[i][j][xxx][yyy]<0:
                    v = torch.zeros((512))
                else:
                    xxx = xxx / 224.
                    xxx = int(xxx * x_mid.size(2))
                    yyy = yyy / 224.
                    yyy = int(yyy * x_mid.size(3))
                    v = x_mid[i, :, xxx, yyy]
                v=v.view(1,1,512).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 512 * 7 * 7)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        x_attribute = x_attribute.view(x_attribute.size(0), 2, self.num_attribute)

        ret['category_output']=x_category
        ret['attr_output']=x_attribute
        ret['feature_output']=x_cat
        return ret

#512
class model_vgg16_all_v3(nn.Module):
    def __init__(self,num_category=50,num_attribute=1000):
        super(model_vgg16_all_v3, self).__init__()
        self.num_category = num_category
        self.num_attribute = num_attribute
        self.vgg16_extractor = VGG16Extractor()

        self.up=nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, 1, 0),
        )
        self.gc1 = GraphConvolution(512, 16)
        self.gc2 = GraphConvolution(16, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048)
        )
        self.cls_4096_512=nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
        )
        self.classifier_category = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_category),
        )
        self.classifier_attribute = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_attribute*2),
        )

    def forward(self, x,landmarks,adj):
        ret={}
        x_lm=self.vgg16_extractor(x)['conv4_3']
        x_lm =self.up(x_lm)
        lm_pos_map = x_lm
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)
        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped.cpu(), dim=2), (pred_h, pred_w))
        # lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        ret['lm_pos_map']=lm_pos_map
        ret['lm_pos_output'] = lm_pos_output


        x_mid =self.vgg16_extractor(x)['pooled_5']
        x1=torch.zeros((1,8,512)).cuda()
        for i in range(x.size(0)):
            vs=torch.zeros((1,1,512)).cuda()
            for j in range(8):
                xxx = lm_pos_output[i][j][0]
                yyy=lm_pos_output[i][j][1]
                if lm_pos_map[i][j][xxx][yyy]<0:
                    v = torch.zeros((512))
                else:
                    xxx = xxx / 224.
                    xxx = int(xxx * x_mid.size(2))
                    yyy = yyy / 224.
                    yyy = int(yyy * x_mid.size(3))
                    v = x_mid[i, :, xxx, yyy]
                v=v.view(1,1,512).cuda()
                vs=torch.cat([vs, v], 1)
            vs=vs[:,1:,:]
            x1=torch.cat([x1,vs],0)
        x1=x1[1:,:,:]
        #print(x1.size())

        x1 = F.relu(self.gc1(x1, adj))
        x1 = F.dropout(x1, 0.5, training=self.training)
        x1 = self.gc2(x1, adj)
        x1=x1.view(x.size(0),8*256)

        x2=self.avgpool(x_mid)
        x2 = x2.view(x.size(0), 512 * 7 * 7)
        x2 = self.classifier2(x2)

        x_cat=torch.cat([x1, x2], 1)
        x_cat=self.cls_4096_512(x_cat)
        x_category=self.classifier_category(x_cat)
        x_attribute=self.classifier_attribute(x_cat)
        x_attribute = x_attribute.view(x_attribute.size(0), 2, self.num_attribute)

        ret['category_output']=x_category
        ret['attr_output']=x_attribute
        ret['features']=x_cat
        return ret

