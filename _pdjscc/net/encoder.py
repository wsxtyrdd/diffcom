from .normalization.GDN import *
from .attention import SNRAttention
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.C = config.C
        self.img_size = config.img_size
        activation = 'prelu'
        device = torch.device('cuda')
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU', prelu='PReLU')
        self.n_downsampling_layers = 2
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu, prelu)

        self.attention1 = SNRAttention(256)
        self.attention2 = SNRAttention(256)
        self.attention3 = SNRAttention(256)
        self.attention4 = SNRAttention(256)
        # self.attention5 = SNRAttention(self.C)

        # (3,32,32) -> (256,16,16), with implicit padding
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=(9, 9), stride=2, padding=(9 - 2) // 2 + 1),
            GDN(256, device, False),
            self.activation(),
        )

        # (256,16,16) -> (256,8,8)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=2, padding=(5 - 2) // 2 + 1),
            GDN(256, device, False),
            self.activation(),
        )

        # (256,8,8) -> (256,8,8)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            GDN(256, device, False),
            self.activation(),
        )

        # (256,8,8) -> (256,8,8)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            GDN(256, device, False),
            self.activation(),
        )

        # (256,8,8) -> (tcn,8,8)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, self.C, kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            GDN(self.C, device, False),
        )

    def forward(self, x, SNR):
        x = self.conv_block1(x)
        x = self.attention1(x, SNR)
        x = self.conv_block2(x)
        x = self.attention2(x, SNR)
        x = self.conv_block3(x)
        x = self.attention3(x, SNR)
        x = self.conv_block4(x)
        x = self.attention4(x, SNR)
        out = self.conv_block5(x)
        # out = self.attention5(x, SNR)
        return out
