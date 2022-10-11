# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

"""STFT-based Loss modules."""


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (Tensor): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    window = window.to(x.device)
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, pad_mode='constant')
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initialize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(STFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(y_mag, x_mag)


class LogMagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(LogMagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        window = self.window.to(x.device)
        mag_x = torch.stft(x, self.fft_size, self.shift_size, self.win_length, window, return_complex=True,
                           pad_mode='constant').abs()
        mag_y = torch.stft(y, self.fft_size, self.shift_size, self.win_length, window, return_complex=True,
                           pad_mode='constant').abs()
        loss = F.mse_loss(torch.log1p(mag_x), torch.log1p(mag_y))

        return loss


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.factor = fft_size / 2048

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss * self.factor, mag_loss * self.factor


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths, factor_sc, factor_mag):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, 'hamming_window')]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


class LossFunc(nn.Module):
    """Loss modules."""

    def _forward_unimplemented(self, *input: Any) -> None:
        raise NotImplementedError("Not implemented function forward in LossFunc")

    def __init__(self, cfg):
        super(LossFunc, self).__init__()

        if cfg['loss_mode'] == 'STFT':
            self.STFTLoss = STFTLoss()
        elif cfg['loss_mode'] == 'MRSTFT':
            self.STFTLoss = MultiResolutionSTFTLoss(cfg)

    def forward(self, outputs, label):
        outputs = outputs.squeeze(1)
        label = label.squeeze(1)
        stft_loss = self.STFTLoss(outputs, label)
        return stft_loss[0] + stft_loss[1]


if __name__ == '__main__':
    import yaml

    cfg_path = r'config/config.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

    outputs = torch.rand([5, 1, 16384])
    label = torch.rand([5, 1, 16384])
    loss_fun = LossFunc(cfg['hparas']['loss'])
    loss = loss_fun(outputs, label)
    pass