from .encoder import *
from .decoder import *
from .discriminator import *
from CommonModules.loss.distortion import Distortion
from .channel import Channel
from random import choice
from CommonModules.loss.distortion import MS_SSIM
from CommonModules.perceptual_similarity.perceptual_loss import PerceptualLoss
from CommonModules.loss import gan_loss
from collections import namedtuple
from functools import partial


def pad_factor(input_image, spatial_dims, factor):
    """Pad `input_image` (N,C,H,W) such that H and W are divisible by `factor`."""

    if isinstance(factor, int) is True:
        factor_H = factor
        factor_W = factor_H
    else:
        factor_H, factor_W = factor

    H, W = spatial_dims[0], spatial_dims[1]
    pad_H = (factor_H - (H % factor_H)) % factor_H
    pad_W = (factor_W - (W % factor_W)) % factor_W
    return F.pad(input_image, pad=(0, pad_W, 0, pad_H), mode='reflect')


class ADJSCC(nn.Module):
    def __init__(self, config):
        super(ADJSCC, self).__init__()
        if config.logger:
            config.logger.info("【Network】: Built Distributed JSCC model, C={}, k/n={}".format(config.C, config.kdivn))

        self.config = config
        self.Encoder = Encoder(config)
        self.Decoder = Decoder(config)
        if config.use_discriminator:
            self.Discriminator = Discriminator(image_dims = config.image_dims, C=config.C)
        self.use_discriminator = config.use_discriminator
        self.channel = Channel(config)
        self.pass_channel = config.pass_channel
        self.MS_SSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        self._lpips = PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(), gpu_ids=[torch.device("cuda:0")])
        self.gan_loss = partial(gan_loss.gan_loss, config.gan_loss_type)
        self.distortion_loss = Distortion(config)
    def feature_pass_channel(self, feature):
        noisy_feature = self.channel(feature)
        return noisy_feature
    def discriminator_forward(self, reconstruction, input_image, latents_quantized, train_generator):
        """ Train on gen/real batches simultaneously. """
        x_gen = reconstruction
        x_real = input_image
        Disc_out = namedtuple("disc_out",
                              ["D_real", "D_gen", "D_real_logits", "D_gen_logits"])

        # Alternate between training discriminator and compression models
        if train_generator is False:
            x_gen = x_gen.detach()

        D_in = torch.cat([x_real, x_gen], dim=0)

        latents = latents_quantized.detach()
        latents = torch.repeat_interleave(latents, 2, dim=0)

        D_out, D_out_logits = self.Discriminator(D_in, latents)
        D_out = torch.squeeze(D_out)
        D_out_logits = torch.squeeze(D_out_logits)

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

        return Disc_out(D_real, D_gen, D_real_logits, D_gen_logits)

    def GAN_loss(self,reconstruction, input_image,latents_quantized,train_generator=False):
        """
        train_generator: Flag to send gradients to generator
        """
        disc_out = self.discriminator_forward(reconstruction, input_image,latents_quantized,train_generator)
        D_loss = self.gan_loss(disc_out, mode='discriminator_loss')
        G_loss = self.gan_loss(disc_out, mode='generator_loss')
        D_gen = torch.mean(disc_out.D_gen).item()
        D_real = torch.mean(disc_out.D_real).item()


        return D_loss, G_loss,D_gen,D_real



    def forward(self, input_sequence,train_generator = True,given_SNR=None):
        B, C, H, W = input_sequence.shape
        if self.training == False:
            n_encoder_downsamples = self.Encoder.n_downsampling_layers
            factor = 2 ** n_encoder_downsamples
            x = pad_factor(input_sequence, input_sequence.size()[2:], factor)
        else:
            x = input_sequence

        if given_SNR is not None:
            self.channel.chan_param = given_SNR
        else:
            random_SNR = choice(self.config.multiple_snr)
            self.channel.chan_param = random_SNR

        SNR = torch.ones([B, 1]).to(x.device) * self.channel.chan_param
        feature = self.Encoder(x, SNR)
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature)
        else:
            noisy_feature = feature
        x_hat = self.Decoder(noisy_feature, SNR)
        if self.training == False:
            x_hat = x_hat[:, :, :H, :W]
            mse_loss = self.distortion_loss(input_sequence, x_hat)
            lpips_loss = self._lpips(input_sequence, x_hat, normalize=True).mean()
            ms_ssim_loss = self.MS_SSIM(input_sequence, x_hat).mean()
            return ms_ssim_loss,mse_loss, lpips_loss, x_hat
        else:
           mse_loss = self.distortion_loss(input_sequence, x_hat)
           lpips_loss = self._lpips(input_sequence,x_hat,normalize=True).mean()
           ms_ssim_loss = self.MS_SSIM(input_sequence, x_hat).mean()
           if self.use_discriminator:
               D_loss, G_loss,D_gen,D_real = self.GAN_loss(x_hat,x,feature,train_generator)
               return ms_ssim_loss,mse_loss,lpips_loss,x_hat,D_loss, G_loss,D_gen,D_real
           else:
               return ms_ssim_loss, mse_loss, lpips_loss, x_hat


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from ADJSCC.config import config

    input_Tensor = torch.ones([2, 3, 256, 256]).cuda()
    model = ADJSCC(config).cuda()
    recon_image, distortion_loss = model(input_Tensor)
    print(recon_image.shape)
