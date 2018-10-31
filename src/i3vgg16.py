import math

import torch
from torch.nn import ReplicationPad3d

from src import inflate


class I3vgg16(torch.nn.Module):
    def __init__(self, vgg2d, frame_nb=16, class_nb=1000, conv_class=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3vgg16, self).__init__()
        
        # lookup = {'conv1_1':'0', 'conv1_2':'2', 'conv2_1':'5', 'conv2_2':'7', 
        #           'conv3_1':'10', 'conv3_2':'12', 'conv3_3':'14', 
        #           'conv4_1':'17', 'conv4_2':'19', 'conv4_3':'21',
        #           'conv5_1':'24', 'conv5_2':'26', 'conv5_3':'28',
        #           'conv6':'31', 'conv7':'33'}

        self.conv1_1 = inflate.inflate_conv(
            vgg2d.features[0], time_dim=3, time_padding=1, center=True)
        self.conv1_2 = inflate.inflate_conv(
            vgg2d.features[2], time_dim=3, time_padding=1, center=True)       
        self.maxpool_1 = inflate.inflate_pool(
            vgg2d.features[4], time_dim=3, time_padding=1, time_stride=2)
        self.conv2_1 = inflate.inflate_conv(
            vgg2d.features[5], time_dim=3, time_padding=1, center=True)       
        self.conv2_2 = inflate.inflate_conv(
            vgg2d.features[7], time_dim=3, time_padding=1, center=True)        
        self.maxpool_2 = inflate.inflate_pool(
            vgg2d.features[9], time_dim=3, time_padding=1, time_stride=2)
        self.conv3_1 = inflate.inflate_conv(
            vgg2d.features[10], time_dim=3, time_padding=1, center=True)        
        self.conv3_2 = inflate.inflate_conv(
            vgg2d.features[12], time_dim=3, time_padding=1, center=True)       
        self.conv3_3 = inflate.inflate_conv(
            vgg2d.features[14], time_dim=3, time_padding=1, center=True)        
        self.maxpool_3 = inflate.inflate_pool(
            vgg2d.features[16], time_dim=3, time_padding=1, time_stride=2)
        self.conv4_1 = inflate.inflate_conv(
            vgg2d.features[17], time_dim=3, time_padding=1, center=True)       
        self.conv4_2 = inflate.inflate_conv(
            vgg2d.features[19], time_dim=3, time_padding=1, center=True)       
        self.conv4_3 = inflate.inflate_conv(
            vgg2d.features[21], time_dim=3, time_padding=1, center=True)        
        self.maxpool_4 = inflate.inflate_pool(
            vgg2d.features[23], time_dim=3, time_padding=1, time_stride=2)
        self.conv5_1 = inflate.inflate_conv(
            vgg2d.features[24], time_dim=3, time_padding=1, center=True)        
        self.conv5_2 = inflate.inflate_conv(
            vgg2d.features[26], time_dim=3, time_padding=1, center=True)        
        self.conv5_3 = inflate.inflate_conv(
            vgg2d.features[28], time_dim=3, time_padding=1, center=True)        
        self.maxpool_5 = inflate.inflate_pool(
            vgg2d.features[30], time_dim=3, time_padding=1, time_stride=2)
        self.fc1 = inflate.inflate_linear(vgg2d.classifier[0], 1)
        
        self.fc2 = inflate.inflate_linear(vgg2d.classifier[3], 1)
        
        self.fc3 = inflate.inflate_linear(vgg2d.classifier[6], 1)
        
        self.relu = torch.nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.maxpool_2(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.maxpool_3(x)
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.maxpool_4(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.maxpool_5(x)
        x_reshape = x.view(x.size(0), -1)
        x = self.fc1(x_reshape)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
