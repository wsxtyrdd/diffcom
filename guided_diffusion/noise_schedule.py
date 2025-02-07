import numpy as np
import torch

from utils.util import Config
from utils import utils_model


class NoiseSchedule:
    def __init__(self, config, logger, device):

        self.cond_config = Config(config.diffcom_series)

        # 1. linear schedule
        betas = np.linspace(config.beta_start, config.beta_end, self.cond_config.num_train_timesteps, dtype=np.float32)

        # 2. cosine schedule
        # t = np.linspace(1, 0, config.num_train_timesteps + 1)[1:]
        # betas = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

        self.betas = torch.from_numpy(betas).to(device)
        self.alphas = 1. - self.betas
        alphas_cumprod = np.cumprod(self.alphas.cpu(), axis=0)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.log_SNRs = torch.log10(self.alphas_cumprod / (1 - self.alphas_cumprod))
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.reduced_alpha_cumprod = torch.div(self.sqrt_1m_alphas_cumprod,
                                               self.sqrt_alphas_cumprod)  # equivalent noise sigma on image
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.num_train_timesteps = self.cond_config.num_train_timesteps
        self.sigma = config.sigma
        self.iter_num = self.cond_config.iter_num

        if not config.CSNR_adapt_t_start:
            self.t_start = self.num_train_timesteps - 1
        else:
            snr = 10 ** (config.CSNR / 10)
            DSNR = np.log10((1 + snr) ** (1 / 48)) * 5
            self.t_start = int((1000 - np.searchsorted(self.log_SNRs.cpu().numpy()[::-1], DSNR)) * config.N)

        logger.info(f'Start from timestep {self.t_start} with SNR from {self.log_SNRs[-1]} to {self.log_SNRs[0]}')

        # create sequence of timestep for sampling
        skip = self.num_train_timesteps // self.iter_num
        if config.skip_type == 'uniform':
            seq = [i * skip for i in np.arange(0, self.t_start // skip)]
        elif config.skip_type == "quad":
            seq = np.sqrt(np.linspace(0, self.num_train_timesteps ** 2, self.t_start))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        self.seq = seq[::-1]

        # plot log-SNR schedule
        # plt.plot(self.log_SNRs.cpu().numpy())
        # plt.xlim(0, 1000)
        # plt.ylim(-15, 15)
        # plt.xlabel('timestep')
        # plt.ylabel('log-SNR')
        # plt.title('log-SNR noise schedule')
        # plt.savefig(os.path.join(config.E_path, 'log_SNR_schedule.png'))
        # plt.close()
