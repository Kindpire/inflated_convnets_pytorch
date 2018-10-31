import math

import torch
from torch.nn import ReplicationPad3d

from src import inflate


class I3vgg(torch.nn.Module):
    def __init__(self, vgg2d, frame_nb=16, class_nb=1000, conv_class=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3vgg, self).__init__()

        self.conv1_1 = inflate.inflate_conv(
            vgg2d.features[0], time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(vgg2d.features[1])
        self.conv1_2 = inflate.inflate_conv(
            vgg2d.features[3], time_dim=3, time_padding=1, center=True)
        self.bn2 = inflate.inflate_batch_norm(vgg2d.features[4])       
        self.maxpool_1 = inflate.inflate_pool(
            vgg2d.features[6], time_dim=3, time_padding=1, time_stride=2)
        self.conv2_1 = inflate.inflate_conv(
            vgg2d.features[7], time_dim=3, time_padding=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(vgg2d.features[8])       
        self.conv2_2 = inflate.inflate_conv(
            vgg2d.features[10], time_dim=3, time_padding=1, center=True)        
        self.bn4 = inflate.inflate_batch_norm(vgg2d.features[11])
        self.maxpool_2 = inflate.inflate_pool(
            vgg2d.features[13], time_dim=3, time_padding=1, time_stride=2)
        self.conv3_1 = inflate.inflate_conv(
            vgg2d.features[14], time_dim=3, time_padding=1, center=True)        
        self.bn5 = inflate.inflate_batch_norm(vgg2d.features[15])
        self.conv3_2 = inflate.inflate_conv(
            vgg2d.features[17], time_dim=3, time_padding=1, center=True)       
        self.bn6 = inflate.inflate_batch_norm(vgg2d.features[18])
        self.conv3_3 = inflate.inflate_conv(
            vgg2d.features[20], time_dim=3, time_padding=1, center=True)        
        self.bn7 = inflate.inflate_batch_norm(vgg2d.features[21])
        self.maxpool_3 = inflate.inflate_pool(
            vgg2d.features[23], time_dim=3, time_padding=1, time_stride=2)
        self.conv4_1 = inflate.inflate_conv(
            vgg2d.features[24], time_dim=3, time_padding=1, center=True)       
        self.bn8 = inflate.inflate_batch_norm(vgg2d.features[25])
        self.conv4_2 = inflate.inflate_conv(
            vgg2d.features[27], time_dim=3, time_padding=1, center=True)       
        self.bn9 = inflate.inflate_batch_norm(vgg2d.features[28])
        self.conv4_3 = inflate.inflate_conv(
            vgg2d.features[30], time_dim=3, time_padding=1, center=True)        
        self.bn10 = inflate.inflate_batch_norm(vgg2d.features[31])
        self.maxpool_4 = inflate.inflate_pool(
            vgg2d.features[33], time_dim=3, time_padding=1, time_stride=2)
        self.conv5_1 = inflate.inflate_conv(
            vgg2d.features[34], time_dim=3, time_padding=1, center=True)        
        self.bn11 = inflate.inflate_batch_norm(vgg2d.features[35])
        self.conv5_2 = inflate.inflate_conv(
            vgg2d.features[37], time_dim=3, time_padding=1, center=True)        
        self.bn12 = inflate.inflate_batch_norm(vgg2d.features[38])
        self.conv5_3 = inflate.inflate_conv(
            vgg2d.features[40], time_dim=3, time_padding=1, center=True)        
        self.bn13 = inflate.inflate_batch_norm(vgg2d.features[41])
        self.maxpool_5 = inflate.inflate_pool(
            vgg2d.features[43], time_dim=3, time_padding=1, time_stride=2)
        self.fc1 = inflate.inflate_linear(vgg2d.classifier[0], 1)
        
        self.fc2 = inflate.inflate_linear(vgg2d.classifier[3], 1)
        
        self.fc3 = inflate.inflate_linear(vgg2d.classifier[6], 1)
        
        self.relu = torch.nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool_2(x)
        x = self.conv3_1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.maxpool_3(x)
        x = self.conv4_1(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.maxpool_4(x)
        x = self.conv5_1(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.maxpool_5(x)
        x_reshape = x.view(x.size(0), -1)
        x = self.fc1(x_reshape)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
