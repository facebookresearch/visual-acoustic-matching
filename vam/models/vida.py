# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import speechmetrics
from torch import Tensor

from vam.models.base_av_model import BaseAVModel


class VIDA(BaseAVModel):
    def __init__(self, args):
        super(VIDA, self).__init__(args)
        self.use_visual = args.use_rgb or args.use_depth
        self.vida = Predictor(input_channel=2, use_rgb=args.use_rgb, use_depth=args.use_depth, no_mask=True,
                              mean_pool_visual=self.use_visual)

        self.first_val = True
        self.best_val_loss = float('inf')
        self.configure_loss()
        self.metrics = speechmetrics.load(['stoi', 'sisdr', 'bsseval', 'mosnet'], window=None)

    def load_weights(self, ckpt):
        self.vida.load_state_dict(ckpt['predictor'])

    def forward(self, audio, rgb, depth):
        inputs = {'spectrograms': audio, 'rgb': rgb, 'depth': depth, 'distance': 0}
        prediction = self.vida.forward(inputs)['pred_mask']

        return prediction

    def dereverb(self, batch, batch_idx):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)
        if self.use_visual:
            rgb, depth = batch['rgb'], batch['depth']
        else:
            rgb, depth = None, None

        # inputs = batch['recv_spec'].permute(0, 3, 1, 2) # use two channel inputs
        recv_spec = self.wav2spec(batch['recv_wav']).permute(0, 3, 1, 2)
        if self.args.decode_wav:
            tgt = batch['src_wav']
            pred = self.forward(recv_spec, rgb, depth)
            pred = torch.cat([torch.exp(pred[:, 0:1, :, :].clone()) - 1, pred[:, 1:2, :, :]], dim=1)
            full_spec = torch.zeros_like(recv_spec, device=self.device)
            full_spec[:, :, :-1, :-1] = pred
            pred = griffinlim(full_spec[:, 0, ...], full_spec[:, 1, ...],
                              torch.hamming_window(400, device=full_spec.device), n_fft=512, hop_length=160,
                              win_length=400, power=1, n_iter=30).to(self.device)
        else:
            tgt = self.wav2spec(batch['src_wav']).permute(0, 3, 1, 2)
            pred = self.forward(recv_spec, rgb, depth)[:, 0, :, :]
            tgt = tgt[:, 0, :, :]
            pred = torch.cat([pred, torch.zeros_like(pred[:, 0:1, :])], dim=1)
            pred = torch.cat([pred, torch.zeros_like(pred[:, :, 0:1])], dim=2)
            assert pred.shape == tgt.shape, (pred.shape, tgt.shape)

            pred = pred.permute(0, 2, 1)
            tgt = tgt.permute(0, 2, 1)

        return {'pred': pred, 'tgt': tgt}


class Predictor(nn.Module):
    def __init__(self, input_channel=2, use_rgb=False, use_depth=False, no_mask=False,
                 limited_fov=False, crop=False, normalize=False, mean_pool_visual=False):
        super(Predictor, self).__init__()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_visual = use_rgb or use_depth
        self.mean_pool_visual = mean_pool_visual

        if self.use_visual:
            if use_rgb:
                self.rgb_net = VisualNet(torchvision.models.resnet18(pretrained=True), 3)
            if use_depth:
                self.depth_net = VisualNet(torchvision.models.resnet18(pretrained=True), 1)
            concat_size = 512 * sum([self.use_rgb, self.use_depth])
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            if self.mean_pool_visual:
                self.conv1x1 = create_conv(concat_size, 512, 1, 0)
            else:
                self.conv1x1 = create_conv(concat_size, 8, 1, 0)  # reduce dimension of extracted visual features

        # if complex, keep input output channel as 2
        self.audio_net = AudioNet(64, input_channel, input_channel, self.use_visual, no_mask,
                                  limited_fov, crop, normalize, mean_pool_visual)
        self.audio_net.apply(weights_init)

    def forward(self, inputs):
        visual_features = []
        if self.use_rgb:
            visual_features.append(self.rgb_net(inputs['rgb']))
        if self.use_depth:
            visual_features.append(self.depth_net(inputs['depth']))
        if len(visual_features) != 0:
            # concatenate channel-wise
            concat_visual_features = torch.cat(visual_features, dim=1)
            concat_visual_features = self.conv1x1(concat_visual_features)
        else:
            concat_visual_features = None

        if self.mean_pool_visual:
            concat_visual_features = self.pooling(concat_visual_features)
        elif len(visual_features) != 0:
            concat_visual_features = concat_visual_features.view(concat_visual_features.shape[0], -1, 1, 1)

        pred_mask, audio_feat = self.audio_net(inputs['spectrograms'], concat_visual_features, inputs['distance'])
        output = {'pred_mask': pred_mask}

        if len(visual_features) != 0:
            audio_embed = self.pooling(audio_feat).squeeze(-1).squeeze(-1)
            visual_embed = concat_visual_features.squeeze(-1).squeeze(-1)
            if self.use_triplet_fc:
                audio_embed = self.audio_triplet_net(audio_embed)
                visual_embed = self.visual_triplet_net(visual_embed)
            if self.use_cos_dist:
                output['audio_feat'] = audio_embed
                output['visual_feat'] = visual_embed
            else:
                output['audio_feat'] = F.normalize(audio_embed, p=2, dim=1)
                output['visual_feat'] = F.normalize(visual_embed, p=2, dim=1)

        return output


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, use_sigmoid=False, use_tanh=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        if use_sigmoid:
            return nn.Sequential(*[upconv, nn.Sigmoid()])
        elif use_tanh:
            return nn.Sequential(*[upconv, nn.Tanh()])
        else:
            return nn.Sequential(*[upconv])


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, use_relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class VisualNet(nn.Module):
    def __init__(self, original_resnet, num_channel=3):
        super(VisualNet, self).__init__()
        original_resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)  # features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x


class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, use_visual=True,
                 no_mask=False, limited_fov=False, crop=False, normalize=False, mean_pool_visual=False):
        super(AudioNet, self).__init__()
        self.use_visual = use_visual
        self.no_mask = no_mask
        self.normalize = normalize

        num_intermediate_feature = 512
        if use_visual:
            if limited_fov:
                num_intermediate_feature += (768 if not mean_pool_visual else 512)
            else:
                num_intermediate_feature += (864 if not mean_pool_visual else 512)

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(num_intermediate_feature,
                                                 ngf * 8)  # 1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        if self.no_mask:
            self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True)
        else:
            # outermost layer use a sigmoid to bound the mask
            self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True, use_sigmoid=True)

    def forward(self, x, visual_feat, locations, distance=None, presence=None, volume=None, material=None,
                room=None):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        upconv_feature_input = audio_conv5feature
        if self.use_visual:
            visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
            visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2],
                                             audio_conv5feature.shape[-1])  # tile visual feature

            if self.normalize:
                visual_feat = F.normalize(visual_feat, p=2, dim=1)
                upconv_feature_input = F.normalize(upconv_feature_input, p=2, dim=1)
            upconv_feature_input = torch.cat((visual_feat, upconv_feature_input), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(upconv_feature_input)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        if self.no_mask:
            prediction = self.audionet_upconvlayer5(
                torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        else:
            prediction = self.audionet_upconvlayer5(
                torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1
        return prediction, audio_conv5feature


def griffinlim(
        specgram: Tensor,
        phase,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: float,
        normalized: bool = False,
        n_iter: int = 32,
        momentum: float = 0.99,
        length: Optional[int] = None,
        rand_init: bool = None
) -> Tensor:
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
        Implementation ported from `librosa`.
    Args:
        specgram (Tensor): A magnitude-only STFT spectrogram of dimension (..., freq, frames)
            where freq is ``n_fft // 2 + 1``.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins
        hop_length (int): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        win_length (int): Window size. (Default: ``n_fft``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
        normalized (bool): Whether to normalize by magnitude after stft.
        n_iter (int): Number of iteration for phase recovery process.
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge.
        length (int or None): Array length of the expected output.
        rand_init (bool): Initializes phase randomly if True, to zero otherwise.
    Returns:
        torch.Tensor: waveform of (..., time), where time equals the ``length`` parameter if given.
    """
    assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
    assert momentum >= 0, 'momentum={} < 0'.format(momentum)

    if normalized:
        warnings.warn(
            "The argument normalized is not used in Griffin-Lim, "
            "and will be removed in v0.9.0 release. To suppress this warning, "
            "please use `normalized=False`.")

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    specgram = specgram.pow(1 / power)

    # randomly initialize the phase
    batch, freq, frames = specgram.size()
    if phase is None:
        if rand_init:
            angles = 2 * math.pi * torch.rand(batch, freq, frames)
        else:
            angles = torch.zeros(batch, freq, frames)
        angles = torch.stack([angles.cos(), angles.sin()], dim=-1) \
            .to(dtype=specgram.dtype, device=specgram.device)
    else:
        # use input phase instead of random phase
        angles = torch.stack([phase.cos(), phase.sin()], dim=-1)
    specgram = specgram.unsqueeze(-1).expand_as(angles)

    # And initialize the previous iterate to 0
    rebuilt = torch.tensor(0.)

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = torch.istft(specgram * angles,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=win_length,
                              window=window,
                              length=length).float()

        # Rebuild the spectrogram
        rebuilt = torch.view_as_real(
            torch.stft(
                input=inverse,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                pad_mode='reflect',
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum / (1 + momentum))
        angles = angles.div(complex_norm(angles).add(1e-16).unsqueeze(-1).expand_as(angles))

    # Return the final phase estimates
    waveform = torch.istft(specgram * angles,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=win_length,
                           window=window,
                           length=length)

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


def complex_norm(
        complex_tensor: Tensor,
        power: float = 1.0
) -> Tensor:
    r"""Compute the norm of complex tensor input.
    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).
    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)
