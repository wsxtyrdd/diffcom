from random import choice

# from loss.distortion import Distortion
import torch
import torch.utils.data
from torch import nn
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
from torch.autograd import Function


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """

    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class AFModule(nn.Module):
    #  Please refer to 'J. Xu, B. Ai, W. Chen et al., “Wireless image transmission using deep source channel coding with attention modules,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 4, pp. 2315–2328, 2021.' for more details.
    def __init__(self, C):
        super(AFModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(C + 1, (C + 1) // 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear((C + 1) // 16, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, SNR):
        feature_pooling = self.avg_pool(x)
        [b, c, _, _] = feature_pooling.shape
        context_information = torch.cat((SNR, feature_pooling.reshape(b, c)), 1)
        scale_factor = self.sigmoid(self.fc2(self.relu1(self.fc1(context_information))))
        out = torch.mul(x, scale_factor.unsqueeze(2).unsqueeze(3))
        return out


class Encoder(nn.Module):
    def __init__(self, C, device, use_attn=True, activation='prelu'):
        super(Encoder, self).__init__()
        self.C = C
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU', prelu='PReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu, prelu)

        if use_attn:
            self.attention1 = AFModule(256)
            self.attention2 = AFModule(256)
            self.attention3 = AFModule(256)
            self.attention4 = AFModule(256)

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
        return out


class Decoder(nn.Module):
    def __init__(self, C, device, use_attn=True, activation='prelu'):
        super(Decoder, self).__init__()
        self.post_pad = nn.ReflectionPad2d(3)
        self.C = C

        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU', prelu='PReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu, prelu)
        self.sigmoid = nn.Sigmoid()

        if use_attn:
            self.attention1 = AFModule(256)
            self.attention2 = AFModule(256)
            self.attention3 = AFModule(256)
            self.attention4 = AFModule(256)
            self.attention5 = AFModule(256)

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


class ADJSCC(nn.Module):
    def __init__(self, C, channel, device):
        super(ADJSCC, self).__init__()
        # if config.logger:
        #     config.logger.info("【Network】: Built ADJSCC model, C={}".format(config.C))
        # self.config = config
        self.device = device
        self.jscc_encoder = Encoder(C, device, use_attn=True)
        self.jscc_decoder = Decoder(C, device, use_attn=True)
        # self.distortion_loss = Distortion(config)
        self.channel = channel

    def feature_pass_channel(self, feature):
        noisy_feature = self.channel(feature)
        return noisy_feature

    def forward(self, input_image, given_SNR=None, mode='ADJSCC'):
        B, C, H, W = input_image.shape
        if given_SNR is not None:
            self.channel.chan_param = given_SNR
        else:
            # used for training
            random_SNR = choice(self.config.multiple_snr)
            self.channel.chan_param = random_SNR

        SNR = torch.ones([B, 1]).to(self.device) * self.channel.chan_param
        feature = self.jscc_encoder(input_image, SNR)
        noisy_feature, _ = self.feature_pass_channel(feature)
        recon_image = self.jscc_decoder(noisy_feature, SNR)
        # distortion_loss = self.distortion_loss.forward(input_image, recon_image)
        return recon_image  # , distortion_loss

    def encode(self, input_image, given_SNR=None):
        B, C, H, W = input_image.shape
        self.channel.chan_param = given_SNR
        SNR = torch.ones([B, 1]).to(self.device) * self.channel.chan_param
        feature = self.jscc_encoder(input_image, SNR)
        return feature

    def decode(self, feature, given_SNR=None):
        B, C, H, W = feature.shape
        SNR = torch.ones([B, 1]).to(self.device) * given_SNR
        recon_image = self.jscc_decoder(feature, SNR)
        return recon_image
