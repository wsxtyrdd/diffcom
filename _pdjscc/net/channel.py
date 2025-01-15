import torch.nn as nn
import numpy as np
import os
import torch


class Channel(nn.Module):
    def __init__(self, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel['type']
        self.chan_param = config.channel['chan_param']  # SNR
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel['type'], config.channel['chan_param']))

    def gaussian_noise_layer(self, input_layer, std):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
        return input_layer + noise

    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def complex_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            # power normalization
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output

    def forward(self, input):
        # input \in R
        channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j
        channel_output = self.complex_forward(channel_in)
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_shape)

        noise = (channel_output - channel_tx).detach()
        noise.requires_grad = False
        channel_rx = channel_tx + noise
        return channel_rx
