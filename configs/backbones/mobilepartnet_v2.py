import torchvision
import torch.nn as nn

class MobilePartNetV2(nn.Module):

    def __init__(self):
        super(MobilePartNetV2, self).__init__()

        # backbone and optimize its architecture
        self.mbnet2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.CNN_feat = self.mbnet2.features
        # resnet.layer4[0].downsample[0].stride = (1,1)
        # resnet.layer4[0].conv2.stride = (1,1)
        #
        # # cnn feature
        # self.mbnet2 = nn.Sequential(
        #     mbnet2.conv1, mbnet2.bn1, mbnet2.relu, mbnet2.maxpool,
        #     mbnet2.layer1, mbnet2.layer2, mbnet2.layer3, mbnet2.layer4)


    def forward(self, x):

        features = self.CNN_feat(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

