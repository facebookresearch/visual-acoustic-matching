# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import speechmetrics
import torch
import numpy as np
import torch.nn.functional as F
import librosa

from vam.models.base_av_model import BaseAVModel
from vam.models.base_av_model import load_rt60_estimator, to_tensor


class BlindReverberator(BaseAVModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = None
        self.configure_loss()
        self.metrics = ['speech_rt60', 'mag_distance', 'log_mag_distance', 'mosnet']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def load_weights(self, ckpt):
        self.load_state_dict(ckpt['state_dict'], strict=False)

    @staticmethod
    def synthesize_ir(rt60, drr):
        rirs = []
        for rt60_est, drr_est in zip(rt60.squeeze().cpu().numpy(), drr.squeeze().cpu().numpy()):
            Fs = 16000
            RT = rt60_est
            DRR = drr_est
            ITDG = 0.01
            dirWidth = 5

            n = np.random.randn(round(RT * Fs))  # base noise
            t = np.arange(len(n)) / Fs
            decayingNoise = n * np.exp(-6.9078 * t / RT)
            decayingNoise = decayingNoise / np.max(np.abs(decayingNoise))

            dirPulse = np.hanning(dirWidth)
            engRev = np.sum(decayingNoise ** 2)  # Reverberant energy
            engDir = engRev * 10 ** (DRR / 10);  # Direct energy, dictated by rev. energy and DRR
            presDir = np.sqrt(engDir / sum(dirPulse ** 2))  # Pressure amplitude for the peak of the direct sound

            rir = np.concatenate(
                [np.zeros(int(0.002 * Fs)), presDir * np.transpose(dirPulse), np.zeros(round(ITDG * Fs)),
                 decayingNoise])  # build the RIR
            rir = rir + np.random.randn(len(rir)) / 1000  # Add some noise (this should be a parameter, not hard coded)
            rir = rir / max(abs(rir))  # Normalized
            rir = np.pad(rir, (0, 32000 - rir.shape[0]))
            rirs.append(rir)

        return to_tensor(rirs)

    def compute_different_source(self, batch):
        rirs = torch.flip(batch['rir_wav'], (1,)).unsqueeze(1)
        # this clip can be any random speech
        content, _ = librosa.load('data/LibriSpeech/dev-clean/1272/128104/1272-128104-0001.flac', sr=16000)
        content = torch.from_numpy(content).to(self.device)
        outputs = []
        for i in range(batch['recv_wav'].shape[0]):
            rir = rirs[i: i + 1]
            audio = content.unsqueeze(0).unsqueeze(0)
            outputs.append(torch.nn.functional.conv1d(audio, rir, padding=rir.shape[-1] - 1))
        outputs = torch.cat(outputs, dim=0).squeeze(1)
        pred_wav = outputs[:, :batch['recv_wav'].shape[1]].detach()
        pred_wav /= pred_wav.abs().max(dim=1, keepdim=True)[0]
        return outputs

    def acoustic_match(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)
        batch['original_src_wav'] = batch['src_wav']

        global RT60_ESTIMATOR
        if RT60_ESTIMATOR is None:
            RT60_ESTIMATOR = load_rt60_estimator(self.device)

        pred_rt60 = self.estimate_rt60(RT60_ESTIMATOR, self.compute_different_source(batch))
        drr = torch.ones_like(pred_rt60) * 2
        if self.args.test_split == 'test-unseen':
            pred_rt60 = torch.ones_like(pred_rt60) * 0.4
        pred_rir = self.synthesize_ir(pred_rt60, drr).to(self.device)
        tgt_wav = batch['recv_wav']

        rirs = torch.flip(pred_rir, (1,)).unsqueeze(1)
        outputs = []
        for i in range(batch['src_wav'].shape[0]):
            rir = rirs[i: i + 1]
            audio = batch['src_wav'].unsqueeze(1)[i:i + 1]
            outputs.append(torch.nn.functional.conv1d(audio, rir, padding=rir.shape[-1] - 1))
        outputs = torch.cat(outputs, dim=0).squeeze(1)
        pred_wav = outputs[:, :batch['recv_wav'].shape[1]].detach()
        pred_wav /= pred_wav.abs().max(dim=1, keepdim=True)[0]

        return {'pred': pred_wav, 'tgt': tgt_wav}