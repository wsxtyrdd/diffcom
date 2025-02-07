"""
Simulate the measurement y = f(x) + n.
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torchvision import torch

from _djscc.network import ADJSCC
from _ntsccp.net.ntscc import CompatibleNTSCC_plus, NTSCC_plus
from channel.channel import Channel
from channel.ofdm_channel import LMMSE_channel_est, LS_channel_est, MMSE_equalization, OFDM, ZF_equalization
from utils.util import Config

# OPERATOR CLASSES -> f(·)

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class NonlinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # H * x
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # H^T * x
        pass

    def ortho_project(self, data, **kwargs):
        # (I - H^T * H) * x
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # (I - H^T * H) * y - H * x
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


def shuffle(x, shuffled_indices=None):
    B, N_s = x.shape
    if shuffled_indices is None:
        shuffled_indices = torch.randperm(N_s)
    x = x.reshape(B, -1)[..., shuffled_indices].reshape(B, N_s)
    return x, shuffled_indices


def de_shuffle(x, shuffled_indices):
    B, N_s = x.shape
    x = x.reshape(B, -1)
    x_rx = torch.zeros_like(x)
    x_rx[..., shuffled_indices] = x
    x_rx = x_rx.reshape(B, N_s)
    return x_rx


class ChannelWrapper(nn.Module):
    def __init__(self, config, logger, device, rescale):
        super(ChannelWrapper, self).__init__()
        self.channel_type = config.channel_type
        self.CSNR = config.CSNR
        self.shuffled_indices = None
        self.rescale = rescale
        if config.channel_type == 'awgn' or config.channel_type == 'rayleigh':
            self.channel = Channel(config.channel_type, config.CSNR, logger, device, rescale=False)
        elif config.channel_type == 'ofdm_tdl':
            self.opt = Config(config.ofdm_tdl)
            self.channel = OFDM(self.opt, device)
        else:
            raise NotImplementedError(f"Channel type {config.channel_type} is not supported.")

    def channel_estimation_wrapper(self, H_t, info_pilot, noise_pwr, M):
        if self.opt.channel_est == 'perfect':
            H_est = H_t.unsqueeze(1)
        elif self.opt.channel_est == 'LS':
            H_est = LS_channel_est(self.channel.pilot, info_pilot)
        elif self.opt.channel_est == 'LMMSE':
            H_est = LMMSE_channel_est(self.channel.pilot, info_pilot, M * noise_pwr)
        return H_est

    def channel_equalization_wrapper(self, H_est, info_sig, M, noise_pwr):
        if self.opt.equalization == 'ZF':
            s_equal = ZF_equalization(H_est, info_sig)
        elif self.opt.equalization == 'MMSE':
            s_equal = MMSE_equalization(H_est, info_sig, M * noise_pwr)
        return s_equal

    def observe(self, s, mask):
        '''
        Transmit the symbols through the channel
        :param s: symbols to be transmitted, shape: [B: batch_size, N_s: num_of_symbols]
        :return:
        '''
        self.s_shape = s.shape
        B, N_s = self.s_shape
        # interleave the symbols to be transmitted
        s, shuffled_indices = shuffle(s)
        mask_sig, _ = shuffle(mask, shuffled_indices)
        self.shuffled_indices = shuffled_indices
        avg_pwr = torch.sum(s ** 2) / mask.sum()
        self.avg_pwr = avg_pwr
        cof_est = None
        cof_gt = None
        if self.channel_type == 'awgn':
            info_sig, channel_usage = self.channel.forward(s, avg_pwr)
            info_sig = info_sig * mask_sig
        # elif self.channel_type == 'rayleigh':
        #     info_sig, H_est, channel_usage = self.channel.forward(s)
        elif self.channel_type == 'ofdm_tdl':
            # for variable rate schemes like NTSCC, N_s is changing with different images
            # we need to pad zeros to s, so that the length of s is a multiple of ofdm_size
            ofdm_size = 2 * self.opt.P * self.opt.S
            if N_s % ofdm_size != 0:
                s = torch.cat([s, torch.zeros(B, ofdm_size - N_s % ofdm_size, device=s.get_device())], dim=-1)
            s_ofdm = s.reshape(B, self.opt.P * 2, self.opt.S, -1)
            M = s_ofdm.shape[-1]
            self.ofdm_shape = s_ofdm.shape
            s_ofdm = s_ofdm[:, :self.opt.P] + 1j * s_ofdm[:, self.opt.P:]
            channel_usage = s_ofdm.numel()
            self.channel.set_pilot(M)
            # pwr = torch.mean(s_ofdm.abs() ** 2, -1, True)
            # s_ofdm = normalize(s_ofdm, 1)

            # note: here s_ofdm is complex, we dont need to multiply 2
            s_ofdm = s_ofdm / torch.sqrt(avg_pwr)
            info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = self.channel(s_ofdm,
                                                                               self.CSNR,
                                                                               cof=None,
                                                                               add_noise=True)
            cof_gt = self.channel.get_cof_from_H(H_t)
            self.noise_pwr = noise_pwr
            if self.opt.blind:
                cof_est = None
            else:
                H_est = self.channel_estimation_wrapper(H_t, info_pilot, noise_pwr, M)
                cof_est = self.channel.get_cof_from_H(H_est)

        return info_sig, cof_est, cof_gt, channel_usage

    def transpose(self, data, cof):
        if self.channel_type == 'ofdm_tdl':
            B, N_s = self.s_shape
            M = self.ofdm_shape[-1]
            s_ofdm_hat = torch.zeros(self.ofdm_shape, device=data.get_device())

            if self.opt.blind:
                #  without channel estimation and equalization
                s_equal = data
            else:
                H_est = torch.fft.fft(cof, dim=-1)
                s_equal = self.channel_equalization_wrapper(H_est, data, M, self.noise_pwr)

            if self.rescale:
                s_equal = s_equal * torch.sqrt(self.avg_pwr)

            s_ofdm_hat[:, :self.opt.P] = torch.real(s_equal)
            s_ofdm_hat[:, self.opt.P:] = torch.imag(s_equal)
            s_hat = s_ofdm_hat.reshape(B, -1)[:, :N_s]
        else:
            if self.rescale:
                data = data * torch.sqrt(self.avg_pwr * 2)
            s_hat = data
        s_hat = de_shuffle(s_hat, self.shuffled_indices)
        return s_hat

    def forward(self, s, mask, cof=None):
        B, N_s = s.shape
        s, _ = shuffle(s, self.shuffled_indices)

        # we assume receiver knows the average power (a float number) of transmitted symbols
        avg_pwr = self.avg_pwr

        if self.channel_type == 'awgn':
            ofdm_sig = s / torch.sqrt(avg_pwr * 2)

        elif self.channel_type == 'ofdm_tdl':
            s = s.reshape(B, self.opt.P * 2, self.opt.S, -1)
            M = s.shape[-1]
            s = s[:, :self.opt.P] + 1j * s[:, self.opt.P:]
            self.channel.set_pilot(M)
            # s = normalize(s, 1)
            s = s / torch.sqrt(avg_pwr)
            # print(s.shape)
            info_pilot, ofdm_sig, H_t, noise_pwr, papr, papr_cp = self.channel(s,
                                                                               self.CSNR,
                                                                               cof=cof,
                                                                               add_noise=False)
        return ofdm_sig


@register_operator(name='djscc')
class DeepJSCC(NonlinearOperator):
    def __init__(self, config, logger, device):
        self.device = device
        self.config = config
        self.channel = ChannelWrapper(config, logger, device, rescale=False)
        self.model = ADJSCC(config.djscc['channel_num'], self.channel, device)
        state_dict = torch.load(config.djscc['jscc_model_path'], map_location=device)
        # map Encoder to jscc_encoder
        for key in list(state_dict.keys()):
            if key.startswith('distortion_loss'):
                state_dict.pop(key)
                continue
            if key.startswith('Encoder'):
                state_dict[key.replace('Encoder', 'jscc_encoder')] = state_dict.pop(key)
            elif key.startswith('Decoder'):
                state_dict[key.replace('Decoder', 'jscc_decoder')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def observe_and_transpose(self, x):
        s = self.encode(x)
        ofdm_sig, cof_est, cof_gt, channel_usage = self.channel.observe(s, torch.ones_like(s))
        s_hat = self.channel.transpose(ofdm_sig, cof_est)
        x_mse = self.decode(s_hat)
        return {"x_mse": x_mse,
                "ofdm_sig": ofdm_sig,
                "s_hat": s_hat,
                "cof_est": cof_est,
                "cof_gt": cof_gt,
                "channel_usage": channel_usage}

    def encode(self, data):
        B, C, H, W = data.shape
        s = self.model.encode(data, given_SNR=self.config.CSNR)
        self.s_shape = s.shape
        # avg_pwr = torch.mean(s ** 2)
        # s = s / torch.sqrt(avg_pwr * 2)
        s = s.reshape(B, -1)
        return s

    def forward(self, s, cof=None):
        # s = self.model.encode(data, given_SNR=self.channel.chan_param)
        ofdm_sig = self.channel.forward(s, cof=cof, mask=torch.ones_like(s))
        return ofdm_sig

    def transpose(self, ofdm_sig, cof=None):
        s_hat = self.channel.transpose(ofdm_sig, cof)
        return s_hat

    def decode(self, s_hat):
        s_hat = s_hat.reshape(self.s_shape)
        x_mse = self.model.decode(s_hat, given_SNR=self.config.CSNR)
        return x_mse


@register_operator(name='ntscc')
class NTSCC(NonlinearOperator):
    def __init__(self, config, logger, device):
        self.device = device
        self.config = config
        self.compatible = config.ntscc['compatible']

        if config.ntscc['compatible']:
            self.ntscc_config = Config({
                'multiple_rate': [1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
                                  240, 256, 272, 288, 304, 320],
                'pretrained': '/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/compatible_NTSCC.pth.tar',
                'eta': config.ntscc['eta'],
                'qp_level': config.ntscc['qp_level']
            })
            self.model = CompatibleNTSCC_plus(self.ntscc_config,
                                              register_channel=False,
                                              qr_anchor_num=6)
        else:
            pretrained_list = ['/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/ckbd2_lmbd_0.013.pth.tar',
                               '/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/ckbd2_lmbd_0.0483.pth.tar',
                               '/media/D/wangsixian/DiffComm/_ntsccp/checkpoints/ckbd2_lmbd_0.18.pth.tar',
                               '/media/D/wangsixian/NTSCC_plus/checkpoint/ckbd2_lmbd_0.36.pth.tar',
                               '/media/D/wangsixian/NTSCC_plus/checkpoint/ckbd2_lmbd_0.72.pth.tar']
            self.ntscc_config = Config({
                'multiple_rate': [1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
                                  240, 256, 272, 288, 304, 320],
                'pretrained': pretrained_list[config.ntscc['qp_level']],
                'eta': config.ntscc['eta'],
                'qp_level': config.ntscc['qp_level']
            })
            self.model = NTSCC_plus(self.ntscc_config, register_channel=False)

        pretrained = torch.load(self.ntscc_config.pretrained, map_location='cpu')
        if 'state_dict' in pretrained:
            pretrained = pretrained['state_dict']
        result_dict = {}
        for key, weight in pretrained.items():
            result_key = key
            if 'attn_mask' not in key and 'rate_adaption_enc.mask' not in key:
                result_dict[result_key] = weight
        print(self.model.load_state_dict(result_dict, strict=False))
        self.model.to(device)
        self.model.eval()

        self.channel = ChannelWrapper(config, logger, device, rescale=True)
        self.indexes = None

    @torch.no_grad()
    def observe_and_transpose(self, x):
        self.indexes = None
        channel_input, mask, indexes = self.encode(x)
        channel_usage = mask.sum().item() / 2
        ofdm_sig, cof_est, cof_gt, _ = self.channel.observe(channel_input, mask)
        s_hat = self.channel.transpose(ofdm_sig, cof_est)
        x_mse = self.decode(s_hat)
        return {"x_mse": x_mse,
                "ofdm_sig": ofdm_sig,
                # "s_hat": s_hat,
                "cof_est": cof_est,
                "cof_gt": cof_gt,
                "channel_usage": channel_usage}

    def encode(self, data):
        B, _, _, _ = data.shape
        # print("data.flatten()[:10]", data.flatten()[:10])
        # print(self.indexes)
        if self.compatible:
            s_masked, mask, indexes = self.model.encode(data,
                                                        self.indexes,
                                                        eta=self.ntscc_config.eta,
                                                        qp_level=self.ntscc_config.qp_level,
                                                        snr=self.config.CSNR)
        else:
            s_masked, mask, indexes = self.model.encode(data,
                                                        self.indexes)
        # print("s_masked.flatten()[:10]", s_masked.flatten()[:10])
        mask = mask.bool()

        # the masked_select will lead to wrong gradient
        # https://github.com/pytorch/pytorch/issues/99638
        # channel_input = torch.masked_select(s_masked, mask).unsqueeze(0)
        # avg_pwr = torch.mean(channel_input ** 2)
        # channel_input = channel_input / torch.sqrt(avg_pwr * 2)

        channel_input = s_masked.reshape(B, -1)
        mask = mask.reshape(B, -1)

        if self.indexes is not None:
            return channel_input
        else:
            self.indexes = indexes
            self.mask = mask
            self.s_masked_shape = s_masked.shape
            return channel_input, mask, indexes

    def forward(self, channel_input, cof=None):
        ofdm_sig = self.channel.forward(channel_input, self.mask, cof=cof)
        return ofdm_sig

    def decode(self, s_masked_hat):
        # s_hat = torch.zeros(self.s_masked_shape).to(s_masked_hat.get_device())
        # s_hat[self.mask] = s_masked_hat
        s_masked_hat = s_masked_hat.reshape(self.s_masked_shape)

        if self.compatible:
            x_mse = self.model.decode(s_masked_hat, self.indexes, qp_level=self.ntscc_config.qp_level,
                                      snr=self.config.CSNR)
        else:
            x_mse = self.model.decode(s_masked_hat, self.indexes)
        return x_mse

    def transpose(self, ofdm_sig, cof=None):
        s_hat = self.channel.transpose(ofdm_sig, cof)
        return s_hat


if __name__ == "__main__":
    import argparse
    import logging
    from torchvision import transforms
    from PIL import Image
    import torchvision
    import numpy as np

    logger = logging.getLogger('test')

    # Create the parser
    config = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add arguments
    config.add_argument('--channel_num', type=int, default=2, help='Number of channels')
    config.add_argument('--channel_type', type=str, default='ofdm_tdl', help='Type of channel')

    # OFDM parameters
    config.add_argument('--P', type=int, default=1, help='OFDM parameter P')
    config.add_argument('--S', type=int, default=8, help='OFDM parameter S')
    config.add_argument('--K', type=int, default=16, help='OFDM parameter K')
    config.add_argument('--L', type=int, default=8, help='OFDM parameter L')
    config.add_argument('--decay', type=int, default=4, help='OFDM parameter decay')
    config.add_argument('--N_pilot', type=int, default=1, help='OFDM parameter N_pilot')
    config.add_argument('--is_clip', type=bool, default=False, help='OFDM parameter is_clip')
    config.add_argument('--channel_est', type=str, default='perfect', help='OFDM parameter channel_est')
    config.add_argument('--equalization', type=str, default='MMSE', help='OFDM parameter equalization')

    # Test ADJSCC and TDL Channel
    # # Parse the arguments
    # config = config.parse_args()
    # config.channel_num = 2
    # config.channel_type = 'ofdm_tdl'
    # config.SNR = 10
    # config.jscc_model_path = '/media/D/wangsixian/DJSCC/checkpoints/PSNR/[PSNR] ADJSCC C=2.pth.tar'
    # operator = DeepJSCC(config, logger, 'cuda')
    # operator.model = operator.model.to('cuda')
    #
    # image_path = '/media/D/wangsixian/DiffComm/testsets/demo_test/69037.png'
    # x = Image.open(image_path).convert('RGB')
    # x = transforms.Resize((256, 256))(x)
    # x = transforms.ToTensor()(x).unsqueeze(0).to(0)
    # x = x.cuda()
    # x = x.repeat(8, 1, 1, 1)
    #
    # x_mse = operator.observe(x, 10)["x_mse"]
    # mse = torch.mean((x - x_mse) ** 2, dim=(1, 2, 3)).cpu().numpy()
    # psnr = 10 * np.log10(1 / mse)
    # print(psnr, psnr.mean())
    # torchvision.utils.save_image(x_mse, 'x_mse.png')

    # Test compatible NTSCC+
    config = config.parse_args()
    config.blind = False
    config.SNR = 10
    config.ntscc = {
        'eta': 0.15,
        'qp_level': 15
    }  # 0 to 100
    config.channel_type = 'awgn'
    operator = NTSCC(config, logger, 'cuda')
    operator.model = operator.model.to('cuda')
    image_path = '/media/D/wangsixian/DiffComm/testsets/demo_test/69037.png'
    x = Image.open(image_path).convert('RGB')
    x = transforms.Resize((256, 256))(x)
    x = transforms.ToTensor()(x).unsqueeze(0).to(0)
    x = x.cuda()
    batch_size = 8
    x = x.repeat(batch_size, 1, 1, 1)
    results = operator.observe_and_transpose(x)
    print(results['ofdm_sig'].flatten()[:40])
    x_mse = results["x_mse"]
    mse = torch.mean((x - x_mse) ** 2, dim=(1, 2, 3)).cpu().numpy()
    psnr = 10 * np.log10(1 / mse)
    print(psnr, psnr.mean())
    channel_usage = results["channel_usage"]
    print(channel_usage / 256 / 256 / 3 / batch_size)
    torchvision.utils.save_image(x_mse, 'x_mse.png')

    s = operator.encode(x)
    ofdm_sig = operator.forward(s)
    print(ofdm_sig.flatten()[:40])
    s_hat = operator.transpose(ofdm_sig)
    x_gluing = operator.decode(s_hat)
    torchvision.utils.save_image(x_gluing, 'x_gluing.png')
