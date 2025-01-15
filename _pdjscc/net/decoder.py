from .normalization.GDN import *
from .attention import SNRAttention
import torch
import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self, config,number_residual = 7):
        super(Decoder, self).__init__()
        self.C = config.C
        self.img_size = config.img_size
        self.post_pad = nn.ReflectionPad2d(3)
        activation = 'prelu'
        device = torch.device('cuda')
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU', prelu='PReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu, prelu)
        self.sigmoid = nn.Sigmoid()

        self.attention1 = SNRAttention(256)
        self.attention2 = SNRAttention(256)
        self.attention3 = SNRAttention(256)
        self.attention4 = SNRAttention(256)
        self.attention5 = SNRAttention(256)



        # (256,8,8) -> (256,8,8)
        self.upconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(self.C, 256, kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            GDN(256, device, True),
            self.activation(),
        )

        # (256,8,8) -> (256,8,8)
        self.upconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            GDN(256, device, True),
            self.activation(),
        )

        # (256,8,8) -> (256,8,8)
        self.upconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            GDN(256, device, True),
            self.activation(),
        )

        self.upconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(5, 5), stride=2, padding=2, output_padding=1),
            GDN(256, device, True),
            self.activation(),
        )

        self.upconv_block5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(9, 9), stride=2, padding=4, output_padding=1),
            GDN(256, device, True),
            self.activation(),
        )

        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(256, 3, kernel_size=(7, 7), stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x, SNR):
        x = self.upconv_block1(x)
        x = self.attention1(x, SNR)
        x = self.upconv_block2(x)
        x = self.attention2(x, SNR)
        x = self.upconv_block3(x)
        x = self.attention3(x, SNR)
        x = self.upconv_block4(x)
        x = self.attention4(x, SNR)
        x = self.upconv_block5(x)
        x = self.attention5(x, SNR)
        out = self.conv_block_out(x)
        return out


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from ADJSCC.config import config

    input_Tensor = torch.ones([2, 16, 64, 64]).cuda()
    SNR = torch.ones([2, 1]).cuda()
    model = Decoder(config).cuda()
    out = model(input_Tensor, SNR)
    print(out.shape)
