import torch
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck

import numpy as np

from .attention import ChannelGate, SpatialGate

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# network architecture based on FAD and different attention modules
class FAD_HAM_Net(nn.Module):
    def __init__(self, num_classes=2, image_shape=(3, 224, 224), pretrain=True, variant='resnet50'):
        super(FAD_HAM_Net, self).__init__()
        self.shape = image_shape

        self.FAD_Head = FAD_Head(self.shape[-1])

        self.FAD_encoder = ResNetEncoder(in_channels=12, variant=variant)

        self.RGB_encoder = ResNetEncoder(in_channels=3, variant=variant)

        self.spa_1 = SpatialGate(kernel_size=5)
        self.spa_2 = SpatialGate(kernel_size=7)
        self.channel_attention = ChannelGate(gate_channels=4096)

        self.downsample = nn.Upsample(size=(14, 14), mode = 'bilinear', align_corners=False)

        if pretrain:
            state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50'])
            load_matched_state_dict(self.FAD_encoder, state_dict, print_stats=True)
            load_matched_state_dict(self.RGB_encoder, state_dict, print_stats=True)

        # produce intermediate feature map for pixel-wise supervion
        self.lastconv1 = nn.Sequential(
            nn.Conv2d(4096 + 1024 + 128, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(), #nn.ReLU()
        )

        # for binary supervision
        self.linear1 = nn.Linear(2048*2, 1024, bias=False)
        self.bn = nn.BatchNorm1d(1024)
        self.drop = nn.Dropout(p=0.6)
        self.cls = nn.Linear(1024, 1, bias=False) #num_classes

    def forward(self, x):
        x0 = x # rgb image 3, 224, 224
        fad = self.FAD_Head(x) # frequency: 12, 224, 224
        fad, fad_3, fad_2, fad_1 = self.FAD_encoder(fad)
        rgb, rgb_3, rgb_2, rgb_1 = self.RGB_encoder(x0)

        # # I commented these lines because It seems that map_x used for training
        # # Uncomment these line if you are going to train the model
        # concate_3 = torch.cat((fad_3, rgb_3), dim=1)
        # concate_2 = torch.cat((fad_2, rgb_2), dim=1)
        # concate_1 = torch.cat((fad_1, rgb_1), dim=1)
        # # high-level feature map using channel attention
        # att_x_3 = self.channel_attention(concate_3) # 14x14, 3
        # att_x_3_14x14 = self.downsample(att_x_3)
        #
        # # low-level feature map using spatial attention
        # att_x_2 = self.spa_1(concate_2) # 28x28, 5
        # att_x_2_14x14 = self.downsample(att_x_2)
        #
        # att_x_1 = self.spa_2(concate_1) # 56x56, 7
        # att_x_1_14x14 = self.downsample(att_x_1)
        #
        # x_concate = torch.cat((att_x_1_14x14, att_x_2_14x14, att_x_3_14x14), dim=1)
        #
        # map_x = self.lastconv1(x_concate)
        # map_x = map_x.squeeze(1)

        x = torch.cat((fad, rgb), dim=1)  # concatenate last output feature
        x = self.linear1(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.cls(x)
        x = torch.sigmoid(x)
        return x


'''
FAD_Head and Filter is based on paper: Thinking in frequency: Face forgery detection by mining frequency-aware clues
The corresponding code is based on an unofficial implementation: https://github.com/yyk-wew/F3Net
'''


class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1 || 0 - 1
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 224, 224]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 224, 224]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 224, 224]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 224, 224]
        return out


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


# ResNet is used as backbone
class ResNetEncoder(ResNet):
    #[3, 4, 6, 3],
    layers = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
    }

    def __init__(self, in_channels=3, variant='resnet50', norm_layer=None):
        super().__init__(
            block=Bottleneck,
            layers=self.layers[variant],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=norm_layer)

        expansion = 4

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        # Delete fully-connected layer
        del self.fc

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56
        x1 = x

        x = self.layer1(x) # 56
        x = self.layer2(x) # 28
        x2 = x

        x = self.layer3(x)
        x = self.layer4(x) #14
        x3 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x, x3, x2, x1


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def _test():
    import torch

    image_x = torch.randn(4, 3, 224, 224)

    fad_ham = FAD_HAM_Net()

    y, map_y = fad_ham(image_x)
    print('binary output shape:', y.shape, 'feature map shape:', map_y.shape)


if __name__ == "__main__":
    _test()
