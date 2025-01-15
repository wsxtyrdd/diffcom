"""
This code started out ported from Mingyu Yang's implementation of the OFDM channel:
https://github.com/mingyuyng/Deep-JSCC-for-images-with-OFDM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clipping(clipping_ratio, x):
    amp = x.abs()
    sigma = torch.sqrt(torch.mean(amp ** 2, -1, True))
    ratio = sigma * clipping_ratio / amp
    scale = torch.min(ratio, torch.ones_like(ratio))

    with torch.no_grad():
        bias = x * scale - x

    return x + bias


def add_cp(x, cp_len):
    if cp_len == 0:
        return x
    else:
        return torch.cat((x[..., -cp_len:], x), dim=-1)


def rm_cp(x, cp_len):
    return x[..., cp_len:]


def batch_conv1d(x, weights):
    '''
    Enable batch-wise convolution using group convolution operations
    x: BxN
    weight: BxL
    '''

    assert x.shape[0] == weights.shape[0]

    b, n = x.shape
    l = weights.shape[1]

    x = x.unsqueeze(0)  # 1xBxN
    weights = weights.unsqueeze(1)  # Bx1xL
    x = F.pad(x, (l - 1, 0), "constant", 0)  # 1xBx(N+L-1)
    out = F.conv1d(x, weight=weights, bias=None, stride=1, dilation=1, groups=b, padding=0)  # 1xBxN

    return out


def PAPR(x):
    power = torch.mean((x.abs()) ** 2, -1)
    pwr_max, _ = torch.max((x.abs()) ** 2, -1)
    return 10 * torch.log10(pwr_max / power)


def normalize(x, power):
    pwr = torch.mean(x.abs() ** 2, -1, True)
    return np.sqrt(power) * x / torch.sqrt(pwr)


def ZF_equalization(H_est, Y):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    return Y / H_est


def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xM
    # Y: NxPxSxM
    # no = complex_multiplication(Y, complex_conjugate(H_est))
    # de = complex_amp2(H_est)**2 + noise_pwr.unsqueeze(-1)
    # return no/de
    no = Y * H_est.conj()
    de = H_est.abs() ** 2 + noise_pwr.unsqueeze(-1)
    return no / de


def LS_channel_est(pilot_tx, pilot_rx):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM
    return torch.mean(pilot_rx, 2, True) / pilot_tx


def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM
    # return complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx))/(1+(noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]))
    return torch.mean(pilot_rx, 2, True) * pilot_tx.conj() / (1 + (noise_pwr.unsqueeze(-1) / pilot_rx.shape[2]))


# Realization of multipath channel as a nn module
class TDL_Channel(nn.Module):
    def __init__(self, opt, device):
        super(TDL_Channel, self).__init__()
        self.opt = opt

        # Generate unit power profile for tdl channel
        power = torch.exp(-torch.arange(opt.L).float() / opt.decay).view(1, 1, opt.L)  # 1x1xL
        self.power = power / torch.sum(power)  # Normalize the path power to sum to 1
        self.device = device

    def sample(self, N, P, M, L):
        # Sample the channel coefficients
        cof = torch.sqrt(self.power / 2) * (torch.randn(N, P, L) + 1j * torch.randn(N, P, L))
        # print("【cof_gt】", cof.shape, cof[..., :2].cpu().numpy())
        cof_zp = torch.cat((cof, torch.zeros((N, P, M - L))), -1)
        H_t = torch.fft.fft(cof_zp, dim=-1).to(self.device)
        return cof, H_t

    def forward(self, input, M, cof=None):
        # Input size:   NxPx(Sx(M+K))
        # Output size:  NxPx(Sx(M+K))
        # Also return the true channel
        # Generate Channel Matrix
        N, P, SMK = input.shape

        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, M, self.opt.L)
        else:
            # cof_zp = torch.cat((cof, torch.zeros((N, P, M - self.opt.L, 2))), 2)
            # cof_zp = torch.view_as_complex(cof_zp)
            H_t = torch.fft.fft(cof, dim=-1)

        signal_real = input.real.float().view(N * P, -1)  # (NxP)x(Sx(M+K))
        signal_imag = input.imag.float().view(N * P, -1)  # (NxP)x(Sx(M+K))

        ind = torch.linspace(self.opt.L - 1, 0, self.opt.L).long()
        cof_real = cof.real[..., ind].view(N * P, -1).float().to(self.device)  # (NxP)xL
        cof_imag = cof.imag[..., ind].view(N * P, -1).float().to(self.device)  # (NxP)xL

        output_real = batch_conv1d(signal_real, cof_real) - batch_conv1d(signal_imag, cof_imag)  # (NxP)x(L+SMK-1)
        output_imag = batch_conv1d(signal_real, cof_imag) + batch_conv1d(signal_imag, cof_real)  # (NxP)x(L+SMK-1)

        output = torch.cat((output_real.view(N * P, -1, 1), output_imag.view(N * P, -1, 1)), -1).view(N, P, SMK,
                                                                                                      2)  # NxPxSMKx2
        output = torch.view_as_complex(output)

        return output, H_t


# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    def __init__(self, opt, device):
        super(OFDM, self).__init__()
        self.opt = opt
        self.device = device
        # Setup the channel layer
        self.channel = TDL_Channel(opt, device)
        self.pilot = None
        self.clip = lambda x: clipping(opt.clip_ratio, x)

    def set_pilot(self, M):
        if self.pilot is None:
            # Generate the pilot signal
            bits = torch.randint(2, (M, 2))
            # torch.save(bits, pilot_path)
            pilot = (2 * bits - 1).float()
            self.pilot = pilot.to(self.device)
            self.pilot = torch.view_as_complex(self.pilot)
            self.pilot = normalize(self.pilot, 1)
            self.pilot_cp = add_cp(torch.fft.ifft(self.pilot), self.opt.K).repeat(self.opt.P, self.opt.N_pilot, 1)
        else:
            pass

    def get_cof_from_H(self, H_t):
        # Get the channel coefficients from the channel matrix
        cof = torch.fft.ifft(H_t, dim=-1)
        return cof

    def forward(self, x, SNR, cof=None, batch_size=None, add_noise=False):
        # Input size: NxPxSxM   The information to be transmitted
        # cof denotes given channel coefficients

        N, P, S, M = x.shape

        # If x is None, we only send the pilots through the channel
        is_pilot = (x == None)

        if not is_pilot:

            # Change to new complex representations
            N = x.shape[0]

            # IFFT:                    NxPxSxM  => NxPxSxM
            x = torch.fft.ifft(x, dim=-1)

            # Add Cyclic Prefix:       NxPxSxM  => NxPxSx(M+K)
            x = add_cp(x, self.opt.K)

            # Add pilot:               NxPxSx(M+K)  => NxPx(S+1)x(M+K)
            self.set_pilot(M)
            pilot = self.pilot_cp.repeat(N, 1, 1, 1)
            x = torch.cat((pilot, x), 2)
            Ns = self.opt.S
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N, 1, 1, 1)
            Ns = 0

        # Reshape:                 NxPx(S+1)x(M+K)  => NxPx(S+1)(M+K)
        x = x.view(N, self.opt.P, (Ns + self.opt.N_pilot) * (M + self.opt.K))

        # PAPR before clipping
        papr = PAPR(x)

        # Clipping (Optional):     NxPx(S+1)(M+K)  => NxPx(S+1)(M+K)
        if self.opt.is_clip:
            x = self.clip(x)

        # PAPR after clipping
        papr_cp = PAPR(x)

        # Pass through the Channel:        NxPx(S+1)(M+K)  =>  NxPx((S+1)(M+K))
        y, H_t = self.channel(x, M, cof=cof)

        # Calculate the power of received signal
        pwr = torch.mean(y.abs() ** 2, -1, True)
        noise_pwr = pwr * 10 ** (-SNR / 10)

        # Generate random noise
        if add_noise:
            noise = torch.sqrt(noise_pwr / 2) * (torch.randn_like(y) + 1j * torch.randn_like(y))
            y_noisy = y + noise
        else:
            y_noisy = y

        # NxPx((S+S')(M+K))  =>  NxPx(S+S')x(M+K)
        output = y_noisy.view(N, self.opt.P, Ns + self.opt.N_pilot, M + self.opt.K)

        y_pilot = output[:, :, :self.opt.N_pilot, :]  # NxPxS'x(M+K)
        y_sig = output[:, :, self.opt.N_pilot:, :]  # NxPxSx(M+K)

        if not is_pilot:
            # Remove Cyclic Prefix:
            info_pilot = rm_cp(y_pilot, self.opt.K)  # NxPxS'xM
            info_sig = rm_cp(y_sig, self.opt.K)  # NxPxSxM

            # FFT:
            info_pilot = torch.fft.fft(info_pilot, dim=-1)
            info_sig = torch.fft.fft(info_sig, dim=-1)

            return info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp
        else:
            info_pilot = rm_cp(y_pilot, self.opt.K)  # NxPxS'xM
            info_pilot = torch.fft.fft(info_pilot, dim=-1)

            return info_pilot, H_t, noise_pwr


if __name__ == "__main__":
    import argparse

    opt = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opt.P = 1
    opt.S = 6
    opt.M = 64
    opt.K = 16
    opt.L = 8
    opt.decay = 4
    opt.N_pilot = 1
    opt.SNR = 10
    opt.is_clip = False

    ofdm = OFDM(opt, 0)

    input_f = torch.randn(128, opt.P, opt.S, opt.M) + 1j * torch.randn(1, opt.P, opt.S, opt.M)
    input_f = normalize(input_f, 1)
    input_f = input_f.cuda()

    info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = ofdm(input_f, opt.SNR, add_noise=True)
    H_t = H_t.cuda()
    print(noise_pwr.shape)

    err = input_f * H_t.unsqueeze(1) - info_sig
    print(f'OFDM path error :{torch.mean(err.abs() ** 2).data}')

    H_est_LS = LS_channel_est(ofdm.pilot, info_pilot)
    err_LS = torch.mean((H_est_LS.squeeze() - H_t.squeeze()).abs() ** 2)
    print(f'LS channel estimation error :{err_LS.data}')

    H_est_LMMSE = LMMSE_channel_est(ofdm.pilot, info_pilot, opt.M * noise_pwr)
    err_LMMSE = torch.mean((H_est_LMMSE.squeeze() - H_t.squeeze()).abs() ** 2)
    print(f'LMMSE channel estimation error :{err_LMMSE.data}')

    print(noise_pwr.shape, info_sig.shape, H_est_LMMSE.shape)

    rx_ZF = ZF_equalization(H_t.unsqueeze(1), info_sig)
    err_ZF = torch.mean((rx_ZF.squeeze() - input_f.squeeze()).abs() ** 2)
    print(f'ZF error :{err_ZF.data}')

    rx_MMSE = MMSE_equalization(H_t.unsqueeze(1), info_sig, opt.M * noise_pwr)
    err_MMSE = torch.mean((rx_MMSE.squeeze() - input_f.squeeze()).abs() ** 2)
    print(f'MMSE error :{err_MMSE.data}')
