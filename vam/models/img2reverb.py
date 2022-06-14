# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import speechmetrics
import torch
import torch.nn.functional as F

from image2reverb.stft import STFT
from image2reverb.mel import LogMel
from vam.models.base_av_model import BaseAVModel
from image2reverb.model import Image2Reverb
from vam.datasets.ss_speech_dataset import to_tensor


class Img2Reverb(BaseAVModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = Image2Reverb(None, None)
        self.configure_loss()
        self.metrics = ['speech_rt60', 'mag_distance', 'log_mag_distance', 'mosnet']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def load_weights(self, ckpt):
        self.load_state_dict(ckpt['state_dict'], strict=False)

    def acoustic_match(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)
        batch['original_src_wav'] = batch['src_wav']

        pred_rir_spec = self.model.forward(torch.cat([batch['rgb'].float(), batch['depth'].float()], dim=1))
        stft = LogMel() if self.model.stft_type == "mel" else STFT()
        pred_rir_spec = pred_rir_spec.permute(0, 1, 3, 2)
        pred_rir = torch.stack([to_tensor(stft.inverse(s.squeeze())) for s in pred_rir_spec], dim=0).to(self.device)
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