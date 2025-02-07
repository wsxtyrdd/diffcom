import numpy as np
import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type, SNR, logger, device, rescale=True):
        super(Channel, self).__init__()
        self.chan_type = channel_type
        self.chan_param = SNR
        self.device = device
        self.logger = logger
        self.rescale = rescale
        if self.logger:
            self.logger.info('【Channel】: Built {} channel, SNR {} dB'.format(
                channel_type, SNR))

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        # fast rayleigh channel
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer), device=device) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer), device=device) ** 2) / np.sqrt(2)
        return input_layer * h + noise, h

    def forward(self, input, avg_pwr=None, power=1):
        B = input.size()[0]
        if avg_pwr is None:
            avg_pwr = torch.mean(input ** 2)
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(B, -1)
        channel_in = channel_in[:, ::2] + channel_in[:, 1::2] * 1j
        channel_usage = channel_in.numel()
        if self.chan_type == 'awgn':
            channel_output = self.channel_forward(channel_in)
            channel_rx = torch.zeros_like(channel_tx.reshape(B, -1))
            channel_rx[:, ::2] = torch.real(channel_output)
            channel_rx[:, 1::2] = torch.imag(channel_output)
            channel_rx = channel_rx.reshape(input_shape)
            if self.rescale:
                return channel_rx * torch.sqrt(avg_pwr * 2), channel_usage
            else:
                return channel_rx, channel_usage
        elif self.chan_type == 'rayleigh':
            channel_output, channel_response = self.channel_forward(channel_in)
            # h = torch.zeros_like(channel_tx.reshape(B, -1))
            # h[:, ::2] = channel_response
            # h[:, 1::2] = channel_response
            # h = h.reshape(input_shape)
            channel_rx = torch.zeros_like(channel_tx.reshape(B, -1))
            channel_rx[:, ::2] = torch.real(channel_output)
            channel_rx[:, 1::2] = torch.imag(channel_output)
            channel_rx = channel_rx.reshape(input_shape)
            if self.rescale:
                return channel_rx * torch.sqrt(avg_pwr * 2), channel_response, channel_usage
            else:
                return channel_rx, channel_response, channel_usage

    def channel_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'noiseless':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output, h = self.rayleigh_noise_layer(channel_tx,
                                                       std=sigma)
        return chan_output, h
