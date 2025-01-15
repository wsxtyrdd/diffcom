import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, image_dims,C, spectral_norm=True):
        """
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = 220 used in [1], C = C_in' if encoder output used
                        as context.

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression",
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()

        self.image_dims = image_dims
        im_channels = self.image_dims[0]
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)

        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.latent = nn.Conv2d(C, context_C_out, kernel_size=3,padding = 1 )
        self.context_upsample = nn.Upsample(scale_factor=4, mode='nearest')

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2,padding = 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.d_conv_head = norm(nn.Conv2d(im_channels + context_C_out, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.d_conv_0 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.d_conv_1 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.d_conv_a = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, stride = 1,padding = 1))

        self.d_conv_b = nn.Conv2d(filters[3], 1, kernel_dim, stride=1,padding = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
              torch.nn.init.normal_(m.weight.data,std=0.02)
              torch.nn.init.constant(m.bias.data, 0.0)

    def forward(self, x, y):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        """
        batch_size = x.size()[0]

        # Concatenate upscaled encoder output y as contextual information
        y = self.activation(self.latent(y))
        y = self.context_upsample(y)

        x = torch.cat((x, y), dim=1)
        x = self.activation(self.d_conv_head(x))
        x = self.activation(self.d_conv_0(x))
        x = self.activation(self.d_conv_1(x))
        x = self.activation(self.d_conv_a(x))
        out_logits = self.d_conv_b(x).view(-1, 1)
        out = torch.sigmoid(out_logits)

        return out, out_logits