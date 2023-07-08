import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RMM import rmm

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class EEM(nn.Module):  # Edge Enhancement Module
    def __init__(self,in_channels,out_channels):
        super(EEM, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3*in_channels,out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels,out_channels)
    def forward(self,x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)

class GRM(nn.Module):  # Global Refinement Module
    def __init__(self):
        super(GRM, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 3, 1, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT2 = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, x, c):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(c), 0.1, inplace=True))
        m = F.leaky_relu(self.SFT2(torch.cat([x, scale], 1)))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(c), 0.1, inplace=True))
        return m + shift

class GeSeNet(nn.Module):
    def __init__(self, output):
        super(GeSeNet, self).__init__()
        output = 1

        # inf -> mri   vis -> ct pet spect

        self.vis_conv = ConvLeakyRelu2d(1, 16)
        self.vis_eem = EEM(16, 32)
        self.vis_rmm = rmm(32, 32)

        self.inf_conv = ConvLeakyRelu2d(1, 16)
        self.inf_eem = EEM(16, 32)
        self.inf_rmm = rmm(32, 32)

        self.sft = GRM()


        self.decoder2 = ConvLeakyRelu2d(96, 32)
        self.decoder1 = ConvBnTanh2d(32, 1)

    def forward(self, image_vis, image_ir):
        x_vis_origin = image_vis[:, : 1]
        x_inf_origin = image_ir

        x_vis_p = self.vis_conv(x_vis_origin)
        x_vis_p1 = self.vis_eem(x_vis_p)
        x_vis_p2 = self.vis_rmm(x_vis_p1)
        x_vis_p2 = x_vis_p2[0]

        x_inf_p = self.inf_conv(x_inf_origin)
        x_inf_p1 = self.inf_eem(x_inf_p)
        x_inf_p2 = self.inf_rmm(x_inf_p1)
        x_inf_p2 = x_inf_p2[0]

        x = torch.cat((x_vis_p2, x_inf_p2), dim=1)

        x = self.sft(x, x_inf_p1)

        x = self.decoder2(torch.cat((x, x_vis_p1), dim=1))
        x = self.decoder1(x)
        return x

# if __name__ == '__main__':
#     net = model_test(1)
#     total = sum([param.nelement() for param in net.parameters()])
#     print("Number of parameters: %.2fM" % (total / 1e6))
#     print(net)