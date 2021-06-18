import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchvision.models.resnet import resnet18, resnet50
from models.variable_width_resnet import resnet18vw
from models.coordconv import CoordConv2d


class CNN4(nn.Module):
    def __init__(self, num_classes=10, in_dims=None, hid_dims=None):
        super(CNN4, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        pooled1 = self.adaptive_avgpool(x).squeeze()

        x = self.maxpool(x)  # 14x14
        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        pooled2 = self.adaptive_avgpool(x).squeeze()

        hidden = x
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        pooled3 = self.adaptive_avgpool(x).squeeze()

        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        final_conv = x
        pooled4 = self.avgpool(final_conv)
        pooled4 = pooled4.view(pooled4.size(0), -1)
        logits = self.fc(pooled4)
        return {
            'final_conv': final_conv,
            'pooled1': pooled1,
            'pooled2': pooled2,
            'pooled3': pooled3,
            'pooled4': pooled4,
            'hidden': hidden,  # for LNL
            'before_logits': pooled4,  # for LNL
            'logits': logits
        }


class BiasedMNISTCNN(nn.Module):
    def __init__(self, num_classes=10, in_dims=None, hid_dims=None):
        super(BiasedMNISTCNN, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 96x96

        x = self.maxpool(x)  # 48x48
        x = self.conv2(x)
        x = self.relu(x)  # 48x48

        hidden = x
        pooled2 = self.adaptive_avgpool(x)

        x = self.conv3(x)
        x = self.relu(x)  # 24x24

        x = self.conv4(x)
        x = self.relu(x)  # 12x12

        # x = self.conv5(x)
        # x = self.relu(x)  # 6x6

        final_conv = x
        pooled4 = self.adaptive_avgpool(final_conv).squeeze()
        # pooled4 = pooled4.view(pooled4.size(0), -1)
        logits = self.fc(pooled4)
        return {
            'final_conv': final_conv,
            'pooled2': pooled2.squeeze(),
            'pooled4': pooled4,
            'hidden': hidden,  # for LNL
            'before_logits': pooled4,  # for LNL
            'logits': logits
        }


class BiasedMNISTCoordConv(nn.Module):
    # https://arxiv.org/pdf/1807.03247.pdf
    def __init__(self, num_classes=10, in_dims=None, hid_dims=None):
        super(BiasedMNISTCoordConv, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = CoordConv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = CoordConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 96x96

        x = self.maxpool(x)  # 48x48
        x = self.conv2(x)
        x = self.relu(x)  # 48x48

        hidden = x
        pooled2 = self.adaptive_avgpool(x)

        x = self.conv3(x)
        x = self.relu(x)  # 24x24

        x = self.conv4(x)
        x = self.relu(x)  # 12x12

        # x = self.conv5(x)
        # x = self.relu(x)  # 6x6

        final_conv = x
        pooled4 = self.adaptive_avgpool(final_conv).squeeze()
        # pooled4 = pooled4.view(pooled4.size(0), -1)
        logits = self.fc(pooled4)
        return {
            'final_conv': final_conv,
            'pooled2': pooled2.squeeze(),
            'pooled4': pooled4,
            'hidden': hidden,  # for LNL
            'before_logits': pooled4,  # for LNL
            'logits': logits
        }


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes, in_dims=None, hid_dims=None, norm_layer=nn.BatchNorm2d, width=64):
        super(ResNetWrapper, self).__init__()
        self.model = self.create_model(width, norm_layer)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(1)

    def create_model(self, width, norm_layer):
        return resnet18vw(width=width, pretrained=False, norm_layer=norm_layer)

    def freeze_layers(self, layers):
        self.layers_to_freeze = layers
        if 'model.conv1' in layers:
            self.model.conv1.eval()
        if 'model.bn1' in layers:
            self.model.bn1.eval()
        if 'model.layer1' in layers:
            self.model.layer1.eval()
        if 'model.layer2' in layers:
            self.model.layer2.eval()

    def train(self, mode):
        super().train(mode)
        if hasattr(self, 'layers_to_freeze'):
            self.freeze_layers(self.layers_to_freeze)

    def forward(self, x):
        x = self.model.conv1(x)  # 64 x 112 x 112 (assuming x = 3x224x224)
        conv1 = x
        pooled_conv1 = self.adaptive_avgpool(conv1).squeeze()
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # 64 x 56 x 56
        x = self.model.layer1(x)  # 64 x 56 x 56
        layer1 = x
        pooled1 = self.adaptive_avgpool(x).squeeze()

        x = self.model.layer2(x)  # 128 x 28 x 28
        layer2 = x
        pooled2 = self.adaptive_avgpool(x).squeeze()

        hidden = self.adaptive_avgpool(x)
        x = self.model.layer3(x)  # 256 x 14 x 14
        layer3 = x
        pooled3 = self.adaptive_avgpool(x).squeeze()

        x = self.model.layer4(x)  # 512 x 7 x 7
        layer4 = x
        x = self.model.avgpool(x)  # 512
        pooled4 = x.squeeze()
        x = torch.flatten(pooled4, 1)
        x = self.model.fc(x)
        return {
            'hidden': hidden,  # for LNL

            'model.conv1': conv1,
            'model.pooled_conv1': pooled_conv1,
            'model.layer1.1.conv2': layer1,
            'model.pooled1': pooled1.squeeze(),

            'model.layer2.1.conv2': layer2,
            'model.layer2_flattened': layer2.squeeze().view(layer2.shape[0], -1),
            'model.pooled2': pooled2.squeeze(),

            'model.layer3.1.conv2': layer3,
            'model.pooled3': pooled3.squeeze(),

            'model.layer4.1.conv2': layer4,
            'model.pooled4': pooled4.squeeze(),
            'model.fc': x,
            'before_logits': pooled4,
            'logits': x
        }


class ResNet18(ResNetWrapper):
    def __init__(self, num_classes, in_dims=None, hid_dims=None, norm_layer=nn.BatchNorm2d, width=64):
        super(ResNet18, self).__init__(num_classes, in_dims, hid_dims, norm_layer, width)


class ResNet10(ResNetWrapper):
    def __init__(self, num_classes, in_dims=None, hid_dims=None, norm_layer=nn.BatchNorm2d, width=32):
        super(ResNet10, self).__init__(num_classes, in_dims, hid_dims, norm_layer, width)


if __name__ == "__main__":
    biased_mnist_x = torch.randn((32, 3, 160, 160))
    model = ResNet10(10)
    model(biased_mnist_x)
