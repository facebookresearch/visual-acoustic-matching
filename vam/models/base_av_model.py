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
from torchaudio.transforms import GriffinLim
from pesq import pesq

from vam.datasets.ss_speech_dataset import normalize, to_tensor
from vam.loss import MultiResolutionSTFTLoss, STFTLoss, LogMagSTFTLoss


RT60_ESTIMATOR = None

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def load_rt60_estimator(device):
    """
    This RT60 estimator was pretrained on SoundSpaces-Speech where GT RIRs are available
    """
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
            else:
                self.loss = LogMagSTFTLoss(fft_size=512, shift_size=self.args.hop_length, win_length=400,
                                     window="hamming_window")
        else:
            # only for spectrogram based solutions
            if self.args.loss == 'mse':
                self.loss = nn.MSELoss()
            elif self.args.loss == 'l1':
                self.loss = nn.L1Loss()
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
            stats['mag_loss'] = loss[1]
            stats['sc_loss'] = loss[0]
            stats['loss'] += loss[0] + loss[1]
        else:
            stats['mag_loss'] = loss
            stats['loss'] += stats['mag_loss']

        return stats, pred, tgt

    def training_epoch_end(self, outputs):
        metrics = outputs[0].keys()
        output_str = f'Train epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.Tensor([output[metric] for output in outputs]).mean()
            self.logger.experiment.add_scalar(f'train/{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

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

    def validation_step(self, batch, batch_idx, test=False):
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
            if metric == 'mag_loss' and avg_value < self.best_val_loss:
                self.best_val_loss = avg_value
                if self.global_rank == 0:
                    model_dir = os.path.join(self.args.model_dir, self.args.version)
                    os.makedirs(model_dir, exist_ok=True)
                    ckpt_path = os.path.join(model_dir, f'best_val.ckpt')
                    torch.save({'state_dict': self.state_dict(),
                                'hparams': self.args}, ckpt_path)

        self.print(output_str[:-2])

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, test=True)

    def test_epoch_end(self, outputs):
        gathered_outputs = self.all_gather(outputs)

        metrics = gathered_outputs[0].keys()
        output_str = f'Test epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.stack([output[metric] for output in gathered_outputs], dim=0).mean()
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