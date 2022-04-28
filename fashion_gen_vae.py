# **Model**

from __future__ import division
from __future__ import print_function, division
import torch.nn as nn
import torch

"""General architecture and standard code adopted from
https://github.com/GlassyWing/nvae and https://github.com/NVlabs/NVAE"""


class Swish(nn.Module):
    """Swish activation function implemented by Vahdat & Kanutz (2020), "NVAE".
        Multiplies x with x after a sigmoid activation.
        Parameters
        ----------
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
        Taken from https://github.com/NVlabs/NVAE
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# SE-Layer after each residual layer as described in NVAE-Paper
class SELayer(nn.Module):
    """Squeeze and excitation layer proposed by Vahdat & Kanutz (2020), "NVAE".
        Parameters
        ----------
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
        Taken from https://github.com/GlassyWing/nvae
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    """KL divergence to compute the proposed residual normal distribution by Vahdat & Kanutz (2020), "NVAE".
        Parameters
        ----------
        delta_mu: mu of the inference model i.e. the approximate posterior of the given level
        delta_log_var: log_var of the inference model i.e. the approximate posterior of the given level
        mu: mu of the generative model
        log_var: log_var of the generative model

        Implementation References
        --------------
        Taken from https://github.com/GlassyWing/nvae
        Added additional mean computation as multidimensional latent variables are used in this implementation
    """
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=1)

    return torch.mean(torch.mean(loss, dim=0))


def sample_z(mu, log_var):
    """Samples z and realizes the reparameterization trick by creating the standard deviation and then
        sampling z as a combination of mu + std * eps
        ----------
        mu: 10x8x8 mean vector
        log_var: 10x8x8 log variance vector

        Implementation References
        --------------
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(mu)
    return mu + std * eps


def kl(mu, log_var):
    """Traditional computation of the kl-divergence
        ----------
        mu: 10x8x8 mean vector
        log_var: 10x8x8 log variance vector

        Implementation References
        --------------
        Taken from https://github.com/GlassyWing/nvae
        Added additional mean computation as multidimensional latent variables are used in this implementation
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)
    return torch.mean(torch.mean(loss, dim=0))


class LatentConvLayers(nn.Module):
    """Latent convolution layer that reduces spatial dimension (number of filters) to 60 and the filter
        size to 16x16 to create the mu_var layer of the inference model of the corresponding level.
        ----------
        in_channels: number of channels (filters) of the corresponding encoder feature map
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
        Inspired by https://github.com/GlassyWing/nvae
        Added additional mean computation as multidimensional latent variables are used in this implementation
        Changed computation of muvar, number of channels and average pooling size
    """
    def __init__(self, in_channels):
        super().__init__()

        self.mu_var = nn.Sequential(
            nn.Conv2d(in_channels, 60, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool2d(16),
            Swish(),
            nn.Conv2d(60, 60, kernel_size=1)
        )

    def forward(self, x):
        muvar = self.mu_var(x)
        return muvar


class LatentUpLayers(nn.Module):
    """Latent upsampling layer that increases spatial dimension (number of filters) and the filter size to the
        number of filters/ filter size at the corresponding level of the decoder.
        ----------
        out_channels: number of channels (filters) of the corresponding decoder feature map
        times_upsampling: the number of times the latent vector needs to be upsampled that the size of the filter
        corresponds to the filter size of the corresponding decoder feature map
        z: 10x8x8 latent vector

        Implementation References
        --------------
    """
    def __init__(self, out_channels, times_upsampling=0):
        super().__init__()
        self.times_upsampling = times_upsampling
        self.z_up = nn.Sequential(
            nn.Conv2d(30, out_channels, kernel_size=3, padding=1),
            Swish()
        )
        self.z_up2 = nn.Sequential(
            # nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2),
            Swish()
        )

    def forward(self, z):
        zUp = self.z_up(z)
        for i in range(self.times_upsampling):
            zUp = self.z_up2(zUp)
        return zUp


class ConvLayer(nn.Module):
    """Simple convolutional block that increases the number of channels and halves the filter size.
        With dilated convolutions, batchnorm and swish activation
        ----------
        in_channels: spatial dimension of encoder feature map before passed through this block
        out_channels: spatial dimension of encoder feature map after passed through this block
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, dilation=2, padding=2),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
    def forward(self, x):
        return self.seq(x)


class EncResidualBlock(nn.Module):
    """Residual block of the encoder according to Vahdat & Kanutz (2020). With Squeeze and Excitation layer
        ,skip connection and dilated convolutions.
        ----------
        channels: spatial dimension of encoder feature map before and after passed through this block
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
        Taken from https://github.com/GlassyWing/nvae
        Changed to dilated convolutions, removed one convolution
    """
    def __init__(self, channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(channels), Swish(),
            nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(channels), Swish(),
            nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2),
            SELayer(channels))

    def forward(self, x):
        return x + self.seq(x)


class Encoder(nn.Module):
    """Complete Encoder of the model. Convolution block followed by residula block
        followed by a latentConvlayer that creates mu and logvar at the corresponding level.
        z is sampled for the final level of the encoder.
        Planar flow is not used here.
        The resulted feature maps of each level are saved to infer into the decoder later.
        ----------
        hidden_dims: list of spatial dimensions of each level
        input_channels: number of channels of the input image
        filter_sizes: list of filter sizes at the different levels
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
        Inspired by https://github.com/GlassyWing/nvae
        Changed the creation of module list, adjusted layers to match
        this architecture, and so on
    """
    def __init__(self, hidden_dims, input_channels, filter_sizes):
        super().__init__()
        conv_layers = []
        res_layers = []
        self.hidden_dims = hidden_dims
        conv_layers.append(ConvLayer(input_channels, hidden_dims[0]))
        res_layers.append(EncResidualBlock(hidden_dims[0]))
        for dim in hidden_dims[:-1]:
            conv_layers.append(ConvLayer(dim, dim * 2))
            res_layers.append(EncResidualBlock(dim * 2))

        self.conv_module_list = nn.ModuleList(conv_layers)
        self.res_module_list = nn.ModuleList(res_layers)
        self.cond_x = LatentConvLayers(hidden_dims[-1])

    def forward(self, x):
        intermediate_feature_maps = []

        #passes image through resnet + conv for each level
        #saves encoder feature maps
        for i in range(len(self.hidden_dims)):
            x = self.res_module_list[i](self.conv_module_list[i](x))
            if i < len(self.hidden_dims) - 1:
                intermediate_feature_maps.append(x)

        #compute z and muvar for the highest level
        muvar = self.cond_x(x)
        mu, log_var = muvar.chunk(2, dim=1)
        z = sample_z(mu, log_var)

        return z, mu, log_var, intermediate_feature_maps


class UpsamplingLayer(nn.Module):
    """Simple upsampling block that decreases the number of channels and doubles the filter size.
        With upsampling nearest neighbour, dilated convolution, batchnorm and swish activation
        ----------
        in_channels: spatial dimension of decoder feature map before passed through this block
        out_channels: spatial dimension of decpder feature map after passed through this block
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

    def forward(self, x):
        return self.seq(x)


class DecResidual(nn.Module):
    """Decoder Residual block of the encoder according to Vahdat & Kanutz (2020). With Squeeze and Excitation layer
        ,skip connection and depthwise convolution.
        ----------
        channels: spatial dimension of encoder feature map before and after passed through this block
        ext: for computation of the number of groups

        Implementation References
        --------------
        Taken from https://github.com/GlassyWing/nvae
        changed number of groups in depthwise convolution
    """
    def __init__(self, channels, ext):
        super().__init__()
        expanded = ext * channels
        self.seq = nn.Sequential(
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.BatchNorm2d(expanded), Swish(),
            nn.Conv2d(expanded, expanded, kernel_size=5, padding=2, groups=expanded),
            nn.BatchNorm2d(expanded), Swish(),
            nn.Conv2d(expanded, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            SELayer(channels))

    def forward(self, x):
        return x + self.seq(x)


class Decoder(nn.Module):
    """Complete decoder as described by Vahdat & Kanutz (2020). Flow inspired by https://github.com/GlassyWing/nvae
        _init_ defines the differen needed layers and creates module lists
        ----------
        hidden_dims: spatial dimension of encoder feature map before and after passed through this block
        input_channels: number of input channels at each level
        filter_sizes: size of filter maps at each level

        Implementation References
        --------------
        Flow taken from https://github.com/GlassyWing/nvae
        Changed to fit changed architecture (model depth, latent vectors, upsampling etc.), changed to the creation of module lists
    """
    def __init__(self, hidden_dims, input_channels, filter_sizes):
        super().__init__()
        upsample_layers = []
        res_layers = []
        latent_layers = []
        latent_layers2 = []
        latent_up = []
        i = 1
        self.hidden_dims = list(reversed(hidden_dims))
        self.filter_sizes = list(reversed(filter_sizes))

        for dim in self.hidden_dims:

            upsample_layers.append(UpsamplingLayer(dim * 2, dim // 2))
            res_layers.append(DecResidual(dim // 2, ext=i * 2))
            latent_up.append(LatentUpLayers(dim, times_upsampling=i - 1))
            if i < len(self.hidden_dims):
                latent_layers.append(LatentConvLayers(dim // 2))
                latent_layers2.append(LatentConvLayers(dim))
            i += 1

        self.up_module_list = nn.ModuleList(upsample_layers)
        self.res_module_list = nn.ModuleList(res_layers)
        self.latent_conv_list = nn.ModuleList(latent_layers)
        self.latent_conv_list2 = nn.ModuleList(latent_layers2)
        self.latent_up_list = nn.ModuleList(latent_up)

        self.prepare_last_z = LatentUpLayers(self.hidden_dims[-1])
        self.output = nn.Sequential(
            nn.Conv2d(self.hidden_dims[-1] // 2, 3, kernel_size=1)
        )

    def forward(self, z, encoder_feature_maps=None):
        """Controls the flow of the decoder model. Upsamples latent variable and concatenates with decoder feature map.
            Performs decoder upsampling followed by decoder residual block and calculates mu and var.
            If training or reconstruction, encoder feature maps are use to comput delta mu and delta log var
            and residual distribution between encoder and decoder z is computed by kl2.
            If only generation, this step is skipped.
            Returns kl losses and the reconstructed image.

            ----------
            z: latent variable of the highes level
            encoder_feature_maps: saved feature maps of the encoder when training/reconstruction

            Implementation References
            --------------
            Flow taken from https://github.com/GlassyWing/nvae
            Changed to fit changed architecture, changed to the creation of module lists
        """
        kl_losses = []
        i = 0
        self.encoder_maps = None
        if encoder_feature_maps is not None:
            self.encoder_maps = list(reversed(encoder_feature_maps))

        #zero-vector to concatenate with highes-level z
        decoder_out = torch.zeros(z.shape[0], self.hidden_dims[0], self.filter_sizes[0], self.filter_sizes[0],
                                  dtype=torch.float32).cuda()


        for layer in self.up_module_list:

            res_layer = self.res_module_list[i]

            # upsample z to match decoder feature map
            z_upsampled = self.latent_up_list[i](z)

            # concat upsampled z and decoder feature map
            conc = torch.cat([decoder_out, z_upsampled], dim=1)

            # decoder residual block
            decoder_out = res_layer(layer(conc))

            #compute muvar for every but the first z
            if i == len(self.up_module_list) - 1:
                break
            muvar = self.latent_conv_list[i](decoder_out)
            mu, log_var = muvar.chunk(2, dim=1)

            # if training, then encoder maps, if not, can be skipped and onl use decoder
            # taken from https://github.com/GlassyWing/nvae
            if self.encoder_maps is not None:
                conc2 = torch.cat([self.encoder_maps[i], decoder_out], dim=1)
                delta_muvar = self.latent_conv_list2[i](conc2)
                delta_mu, delta_log_var = delta_muvar.chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            z = sample_z(mu, log_var)
            i += 1

        x_recon = torch.tanh(self.output(decoder_out))

        return kl_losses, x_recon




class MY_VAE(nn.Module):
    """Realizes the whole model. Defines encoder, decoder, sampling and the complete control flow.
        ----------
        input_channels: number of channels of the initial input image
        batch_size: number of images per patch
        img_size: size of the images
        hidden_dims: spatial dimension of each level of the encoder and decoder

        Implementation References
        --------------
        Flow taken from https://github.com/GlassyWing/nvae
        Changed to fit changed architecture, changed to the creation of module lists, changed computation of kl loss
    """
    def __init__(self, input_channels=3, batch_size=64, img_size=64, hidden_dims=[128, 256, 512],
                 flow=None):
        super(MY_VAE, self).__init__()
        self.flow = flow
        self.input_channels = input_channels
        self.output_channels = input_channels
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.img_size = img_size
        self.hidden_dim = int(self.hidden_dims[-1] * pow(img_size / pow(2, len(self.hidden_dims)), 2))
        self.filter_sizes = []
        filter_size = self.img_size

        #computation of filter sizes for each level
        for dim in hidden_dims:
            filter_size = filter_size // 2
            self.filter_sizes.append(filter_size)
        self.encoder = Encoder(hidden_dims=self.hidden_dims, input_channels=self.input_channels,
                               filter_sizes=self.filter_sizes)

        self.decoder = Decoder(self.hidden_dims, self.input_channels, self.filter_sizes)
        self.count = 0

    def sample_z(self, mu, log_var):
        """Realizes the reparameterization trick. Samples z using the mu, log_var layer and random epsilon
            ----------
            mu: 10x8x8 mean layer
            log_var: 10x8x8 log variance layer
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """Realizes the flow of the model and computes the final kld loss as a summation of the intermediate
            kld losses.
            ----------
            x: input image
        """
        z, mu, log_var, intermediate_feature_maps = self.encode(x)

        kl_loss = kl(mu, log_var)

        kl_losses, x_recon = self.decode(z, intermediate_feature_maps)

        #used for training information
        # print("main_mmd: {}, kl: {}; kl[0]: {}; 1: {}; 2: {}".format(main_mmd_loss, kl_loss, kl_losses[0], kl_losses[1], kl_losses[2]))
        # if self.count % 10 == 1:
        #     print("kl: {}; kl[0]: {}; 1: {}; 2: {}".format(kl_loss.mean(), kl_losses[0], kl_losses[1], kl_losses[2]))
        #self.count += 1

        #Changed the weighting of the single terms to achieve better training stability
        kld_loss = [kl_loss, kl_losses[0], kl_losses[1]]

        return x_recon, z, mu, log_var, kld_loss

    def encode(self, x):
        z, mu, log_var, intermediate_feature_maps = self.encoder(x)
        return z, mu, log_var, intermediate_feature_maps

    def decode(self, z, encoder_feature_maps=None):
        kl_losses, x_recon = self.decoder(z, encoder_feature_maps)
        return kl_losses, x_recon

        return recon_x