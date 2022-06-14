# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
import json

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
from scipy.signal import fftconvolve
from pyroomacoustics.experimental.rt60 import measure_rt60
from torchaudio.transforms import GriffinLim
from torch.optim.lr_scheduler import LambdaLR
from pesq import pesq

from vam.datasets.ss_speech_dataset import calculate_drr_energy_ratio, normalize, to_tensor
from vam.loss import MultiResolutionSTFTLoss, STFTLoss, LogMagSTFTLoss, MultiResolutionNoSCSTFTLoss


RT60_ESTIMATOR = None

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def compute_reverb(batch, pred_rir):
    outputs = []
    for src, rir in zip(batch['src_wav'], pred_rir):
        src = src.cpu().numpy()
        rir = rir.detach().cpu().numpy()
        outputs.append(normalize(fftconvolve(src, rir)))
    outputs = torch.tensor(outputs, device=pred_rir.device)

    tgt_wav = batch['recv_wav'].to(pred_rir.device)

    return outputs, tgt_wav


def load_rt60_estimator(device):
    from vam.models.visual_net import VisualNet
    rt60_estimator = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
    pretrained_weights = 'data/pretrained-models/rt60_estimator.pth'
    rt60_estimator.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['predictor'])
    rt60_estimator.to(device=device).eval()
    return rt60_estimator


class BaseAVModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args

        self.first_val = True
        self.best_val_loss = float('inf')
        self.test_stats = defaultdict(list)

    def configure_loss(self):
        if self.args.decode_wav:
            if self.args.multires_stft:
                self.loss = MultiResolutionSTFTLoss(fft_sizes=[int(x) for x in self.args.fft_sizes.split(',')],
                                                    hop_sizes=[int(x) for x in self.args.hop_sizes.split(',')],
                                                    win_lengths=[int(x) for x in self.args.win_lengths.split(',')],
                                                    factor_sc=1, factor_mag=1)
            elif self.args.multires_stft_no_sc:
                self.loss = MultiResolutionNoSCSTFTLoss(fft_sizes=[int(x) for x in self.args.fft_sizes.split(',')],
                                                        hop_sizes=[int(x) for x in self.args.hop_sizes.split(',')],
                                                        win_lengths=[int(x) for x in self.args.win_lengths.split(',')])
            elif self.args.spectral_stft:
                self.loss = STFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400, window="hamming_window")
            else:
                self.loss = LogMagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                     window="hamming_window")
        else:
            # only for spectrogram based solutions
            if self.args.loss == 'mse':
                self.loss = nn.MSELoss()
            elif self.args.loss == 'l1':
                self.loss = nn.MSELoss()
            else:
                raise ValueError

    def load_weights(self, ckpt):
        self.load_state_dict(ckpt['state_dict'], strict=False)

    def print(self, *args, **kwargs):
        if self.global_rank == 0:
            logging.info(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)[0]

    def run_step(self, batch, batch_idx, phase=None):
        stats = {'loss': 0}
        if self.args.acoustic_matching:
            output = self.acoustic_match(batch, batch_idx, phase)
        else:
            output = self.dereverb(batch, batch_idx)
        pred, tgt = output['pred'], output['tgt']

        loss = self.loss(pred, tgt)
        if isinstance(loss, tuple):
            stats['pred_loss'] = loss[1]
            stats['sc_loss'] = loss[0]
            stats['loss'] += loss[0] + loss[1]
        else:
            stats['pred_loss'] = loss
            stats['loss'] += stats['pred_loss']

        return stats, pred, tgt

    def training_epoch_end(self, outputs):
        metrics = outputs[0].keys()
        output_str = f'Train epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.Tensor([output[metric] for output in outputs]).mean()
            self.logger.experiment.add_scalar(f'train/{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

        if self.args.scheduler != 'none':
            self.logger.experiment.add_scalar('train/lr', self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0],
                                              self.current_epoch)

        self.print(output_str[:-2])

    def estimate_rt60(self, estimator, wav):
        stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, window=torch.hamming_window(400, device=self.device),
                          pad_mode='constant', return_complex=True)
        spec = torch.log1p(stft.abs()).unsqueeze(1)
        with torch.no_grad():
            estimated_rt60 = estimator(spec)
        return estimated_rt60

    def spec2wav(self, spec):
        if self.args.predict_mask:
            assert self.args.use_real_imag
            assert not (self.args.log1p or self.args.log10)
            wav = torch.istft(spec.permute(0, 3, 2, 1), n_fft=512, hop_length=self.args.hop_length, win_length=400,
                              window=torch.hamming_window(400, device=spec.device))
        else:
            assert spec.shape[1] == 1
            if self.args.log1p:
                spec = torch.exp(spec) - 1
            if self.args.log10:
                spec = torch.pow(10, spec)
            wav = GriffinLim(n_fft=512, hop_length=self.args.hop_length, win_length=400, window_fn=torch.hamming_window,
                             power=1, rand_init=False)(spec.squeeze(1).cpu().permute(0, 2, 1)).to(device=self.device)
        return wav

    def wav2spec(self, wav):
        if self.args.use_real_imag:
            spec = torch.stft(wav, n_fft=512, hop_length=self.args.hop_length, win_length=400,
                              window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                              return_complex=False)
        else:
            stft = torch.stft(wav, n_fft=512, hop_length=self.args.hop_length, win_length=400,
                              window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                              return_complex=True)
            mag = stft.abs()
            phase = stft.angle()

            assert not (self.args.log1p and self.args.log10)
            if self.args.log1p:
                mag = torch.log1p(mag)
            if self.args.log10:
                mag = torch.log10(torch.clamp(mag, min=1e-5))
            spec = torch.stack([mag, phase], dim=-1)

        return spec

    def validation_step(self, batch, batch_idx, convolve_rir=False, test=False):
        stats, pred, tgt = self.run_step(batch, batch_idx, phase='val' if not test else 'test')
        if not test and (self.current_epoch + 1) % self.args.val_freq != 0:
            return stats

        if self.args.eval_input:
            pred_wav = batch['src_wav'] if self.args.acoustic_matching else batch['recv_wav']
        else:
            pred_wav = pred if self.args.decode_wav else self.spec2wav(pred)
        tgt_wav = batch['recv_wav'] if self.args.acoustic_matching else batch['src_wav']

        if self.args.eval_speech_rt60:
            global RT60_ESTIMATOR
            if RT60_ESTIMATOR is None:
                RT60_ESTIMATOR = load_rt60_estimator(self.device)
            pred_rt60 = self.estimate_rt60(RT60_ESTIMATOR, pred_wav)
            tgt_rt60 = self.estimate_rt60(RT60_ESTIMATOR, tgt_wav)
            stats['speech_rt60'] = (pred_rt60 - tgt_rt60).abs().mean().cpu().numpy()
            if test:
                self.test_stats['pred_rt60'] += pred_rt60.cpu().numpy().tolist()
                self.test_stats['tgt_rt60'] += tgt_rt60.cpu().numpy().tolist()

        if 'stft_distance' in self.metrics:
            pred_spec, tgt_spec = eval_stft(pred_wav), eval_stft(tgt_wav)
            stats['stft_distance'] = F.mse_loss(pred_spec, tgt_spec)

        if 'mag_distance' in self.metrics:
            pred_spec, tgt_spec = eval_mag(pred_wav), eval_mag(tgt_wav)
            stats['mag_distance'] = F.mse_loss(pred_spec, tgt_spec)

        if 'log_mag_distance' in self.metrics:
            pred_spec, tgt_spec = eval_mag(pred_wav, log=True), eval_mag(tgt_wav, log=True)
            stats['log_mag_distance'] = F.mse_loss(pred_spec, tgt_spec)

        tgt_wav, pred_wav = tgt_wav.cpu().numpy(), pred_wav.cpu().numpy()
        if self.args.acoustic_matching and self.args.match_rir:
            # calculate t60 and drr for evaluating rir synthesis
            stats['rt60_error'] = np.mean([calculate_relative_rt60_diff(y, x) for x, y in zip(tgt_wav, pred_wav)])

            # calculate drr
            stats['drr_error'] = np.mean(
                [abs(calculate_drr_energy_ratio(x, 0.003) - calculate_drr_energy_ratio(y, 0.003))
                 for x, y in zip(tgt_wav, pred_wav)])

        if not (self.args.match_rir and not convolve_rir):
            if self.args.match_rir and convolve_rir:
                pred_wav, tgt_wav = compute_reverb(batch, to_tensor(pred_wav))
                tgt_wav, pred_wav = tgt_wav.numpy(), pred_wav.numpy()

            # calculate speech measurements
            metric_scores = defaultdict(list)
            for speech_metric in self.speech_metric_names:
                metric_scores[speech_metric] = list()
            if 'pesq' in self.metrics:
                metric_scores['pesq'] = list()
            for i in range(tgt_wav.shape[0]):
                if 'pesq' in self.metrics:
                    try:
                        pesq_score = pesq(16000, tgt_wav[i], pred_wav[i], 'wb')
                    except Exception as e:
                        self.print(e)
                        pesq_score = 0
                    metric_scores['pesq'].append(pesq_score)
                if len(self.speech_metric_names) != 0:
                    scores = self.speech_metrics(pred_wav[i], tgt_wav[i], rate=16000)
                    tgt_scores = self.speech_metrics(tgt_wav[i], pred_wav[i], rate=16000)
                    for metric_name in scores:
                        metric_scores[metric_name].append(scores[metric_name])
                        metric_scores[metric_name+'_diff'].append(abs(scores[metric_name] - tgt_scores[metric_name]))
                        metric_scores[metric_name+'_rel'].append(abs(scores[metric_name] - tgt_scores[metric_name]) / tgt_scores[metric_name])
                        # self.test_stats[metric_name+'_diff'].append(abs(scores[metric_name] - tgt_scores[metric_name]))
            for metric_name, metric_value_list in metric_scores.items():
                stats[metric_name] = to_tensor(np.mean(metric_value_list))

        return stats

    def validation_epoch_end(self, outputs):
        if self.first_val:
            self.first_val = False
            return

        gathered_outputs = self.all_gather(outputs)
        metrics = gathered_outputs[0].keys()
        output_str = f'Val epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.stack([output[metric] for output in gathered_outputs], dim=0).mean()
            self.logger.experiment.add_scalar(f'val/{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

            # manually save the best model
            if metric == 'pred_loss' and avg_value < self.best_val_loss:
                self.best_val_loss = avg_value
                if self.global_rank == 0:
                    model_dir = os.path.join(self.args.model_dir, self.args.version)
                    os.makedirs(model_dir, exist_ok=True)
                    ckpt_path = os.path.join(model_dir, f'best_val.ckpt')
                    torch.save({'state_dict': self.state_dict(),
                                'hparams': self.args}, ckpt_path)

        self.print(output_str[:-2])

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, convolve_rir=True, test=True)

    def test_epoch_end(self, outputs):
        gathered_outputs = self.all_gather(outputs)

        metrics = gathered_outputs[0].keys()
        output_str = f'Test epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.stack([output[metric] for output in gathered_outputs], dim=0).mean()
            tokens = self.args.from_pretrained.split('=')
            if self.args.test_all and len(tokens) > 1:
                ckpt = int(tokens[-1][:tokens[-1].find('.')])
                self.logger.experiment.add_scalar(f"{self.args.test_split}{'_clean' if self.args.eval_clean else ''}/{metric}", avg_value, ckpt)
            output_str += f'{metric}: {avg_value:.4f}, '
        self.print(output_str[:-2])

    def save_test_stats(self):
        postfix = f'_{self.args.test_split}'
        postfix += '_clean' if self.args.eval_clean else ''
        postfix += '_input' if self.args.eval_input else ''
        file_path = os.path.join(self.args.model_dir, self.args.version, 'test_stats' + postfix + '.json')
        with open(file_path, 'w') as fo:
            json.dump(self.test_stats, fo)

    def configure_optimizers(self):
        opt_dict = {'optimizer': torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)}

        num_warmup_step = 10
        if self.args.scheduler == 'constant':
            opt_dict['lr_scheduler'] = {
                        'scheduler': get_constant_schedule_with_warmup(opt_dict['optimizer'], num_warmup_step),
                        'interval': 'epoch',
                        'frequency': 1
                    }
            self.print('Using constant schedule with warmup')
        elif self.args.scheduler == 'linear':
            opt_dict['lr_scheduler'] = {
                        'scheduler': get_linear_schedule_with_warmup(opt_dict['optimizer'], num_warmup_step, self.args.max_epochs),
                        'interval': 'epoch',
                        'frequency': 1
                    }
            self.print('Using linear schedule with warmup')
        elif self.args.scheduler == 'none':
            pass
        else:
            raise ValueError

        return opt_dict


def eval_stft(wav):
    spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=False)

    return spec


def eval_mag(wav, log=False):
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                      return_complex=True)
    if log:
        mag = torch.log(stft.abs() + 1e-5)
    else:
        mag = stft.abs()

    return mag


def calculate_relative_rt60_diff(est, gt):
    rt60_est = measure_rt60_wrapper(est, 16000, 30)
    rt60_gt = measure_rt60_wrapper(gt, 16000, 30)
    diff = abs(rt60_gt - rt60_est)
    rel_diff = diff / rt60_gt
    return rel_diff


def measure_rt60_wrapper(x, sr=16000, decay_db=30):
    rt60 = -1
    while rt60 == -1:
        try:
            rt60 = measure_rt60(x, sr, decay_db)
        except ValueError:
            if decay_db > 10:
                decay_db -= 10
            else:
                rt60 = x.shape[0]

    return rt60
