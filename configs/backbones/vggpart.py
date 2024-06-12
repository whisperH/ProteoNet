import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGPart(nn.Module):

    def __init__(self, **kwargs):
        super(VGGPart, self).__init__()

        # attributes
        # backbone and optimize its architecture
        vgg = torchvision.models.vgg16(pretrained=True)
        vgg.layer4[0].downsample[0].stride = (1,1)
        vgg.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            vgg.conv1, vgg.bn1, vgg.relu, vgg.maxpool,
            vgg.layer1, vgg.layer2, vgg.layer3, vgg.layer4)


    def forward(self, x):

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

