# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
import logging
import numpy as np
import torchvision
import speechmetrics

from vam.models.my_conformer import Conformer
from vam.models.vit import ViT
from vam.models.base_av_model import BaseAVModel

DEREVERBERATOR = None


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def update_args(args, dic):
    for key, value in dic.items():
        setattr(args, key, value)
    return args


def load_dereverberator(device, use_vida=False):
    if use_vida:
        from vam.models.vida import VIDA
        from vam.trainer import parser
        pretrained_weights = 'data/pretrained-models/dereverb/vida.pth'
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        default_args = parser.parse_args("")
        update_args(default_args, {'use_rgb': False, 'use_depth': False, 'log1p': True, 'decode_wav': True, 'dereverb': True})
        dereverberator = VIDA(default_args).to(device=device)
        dereverberator.load_weights(state_dict)
        dereverberator.eval()
    else:
        from vam.models.generative_avitar import GenerativeAViTAr
        from vam.trainer import parser
        pretrained_weights = 'data/pretrained-models/dereverb/avitar.pth'
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        default_args = parser.parse_args("")
        update_args(default_args, state_dict['hyper_parameters'])
        dereverberator = GenerativeAViTAr(default_args).to(device=device)
        dereverberator.load_state_dict(state_dict['state_dict'])
        dereverberator.eval()
    return dereverberator


class AViTAr(BaseAVModel):
    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters(args)
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args

        self.use_visual = args.use_cnn
        if args.use_rgb or args.use_depth:
            assert self.use_visual

        num_channel = args.use_rgb * 3 + args.use_depth
        if args.use_cnn:
            conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers = list(torchvision.models.resnet18(pretrained=args.pretrained_cnn).children())[1:-2]
            self.cnn = nn.Sequential(conv1, *layers)

        crossmodal_dropout_p = input_dropout_p = feed_forward_dropout_p = attention_dropout_p = conv_dropout_p = 0 if args.no_dropout else args.dropout
        crossmodal_dropout_p = args.crossmodal_dropout_p if args.crossmodal_dropout_p != -1 else crossmodal_dropout_p
        input_dim = 257 if not self.args.encode_wav else args.ngf * (2 ** len(self.args.encoder_ratios.split(',')))
        self.conformer = Conformer(input_dim=input_dim, num_encoder_layers=args.num_encoder_layers,
                                   input_dropout_p=input_dropout_p, feed_forward_dropout_p=feed_forward_dropout_p,
                                   attention_dropout_p=attention_dropout_p, conv_dropout_p=conv_dropout_p,
                                   use_crossmodal_layer=self.use_visual, conv_kernel_size=args.conv_kernel_size, decode_wav=args.decode_wav,
                                   encode_wav=args.encode_wav, encoder_ratios=args.encoder_ratios,
                                   decoder_ratios=args.decoder_ratios, encoder_residual_layers=args.encoder_residual_layers,
                                   decoder_residual_layers=args.decoder_residual_layers,
                                   ngf=args.ngf, use_visual_pe=self.args.use_visual_pe, crossmodal_dropout_p=crossmodal_dropout_p,
                                   )

        self.first_val = True
        self.best_val_loss = float('inf')
        self.configure_loss()
        self.metrics = ['speech_rt60', 'mag_distance', 'log_mag_distance', 'mosnet']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def get_visual_feat(self, batch):
        visual_input = []
        if self.args.use_depth:
            visual_input.append(batch['depth'])
        if self.args.use_rgb:
            visual_input.append(batch['rgb'])
        visual_input = torch.cat(visual_input, dim=1)

        if self.args.use_cnn:
            img_feat = self.cnn(visual_input)
            if self.args.adaptive_pool:
                tgt_shape = (1, 1) if self.args.mean_pool else (img_feat.shape[-2]//2, img_feat.shape[-1]//2)
                img_feat = F.adaptive_avg_pool2d(img_feat, tgt_shape)
            img_feat = img_feat.reshape(img_feat.size(0), img_feat.size(1), -1).permute(0, 2, 1)
        else:
            img_feat = None

        return img_feat

    def print(self, string):
        if self.global_rank == 0:
            logging.info(string)

    def acoustic_match(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        if self.args.dereverb_avspeech:
            dereverb_avspeech(batch, self.args.encode_wav, self.device, use_vida=self.args.use_vida)
        batch['original_src_wav'] = batch['src_wav']

        if self.args.convolve_random_rir:
            convolve_random_rir(batch)

        if self.args.use_audio_da:
            apply_audio_data_augmentation(self.args, batch, phase)

        if self.args.decode_wav:
            if self.args.encode_wav:
                src = batch['src_wav']
                tgt = batch['recv_wav']
            else:
                src = self.wav2spec(batch['src_wav'])[..., 0].permute(0, 2, 1)
                tgt = torch.nn.functional.pad(batch['recv_wav'], (0, 128))
        else:
            src = self.wav2spec(batch['src_wav'])[..., 0].permute(0, 2, 1)
            tgt = self.wav2spec(batch['recv_wav'])[..., 0].permute(0, 2, 1)

        img_feat = self.get_visual_feat(batch) if self.use_visual else None
        pred, pred_rt60 = self.conformer(src, img_feat=img_feat)
        if self.args.decode_wav:
            pred = pred.squeeze(1)

        return {'pred': pred, 'tgt': tgt, 'pred_rt60': pred_rt60}

    def dereverb(self, batch, batch_idx):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        if self.args.decode_wav:
            if self.args.encode_wav:
                src = batch['recv_wav']
                tgt = batch['src_wav']
            else:
                src = self.wav2spec(batch['recv_wav'])[..., 0].permute(0, 2, 1)
                tgt = torch.nn.functional.pad(batch['src_wav'], (0, 128))
        else:
            src = self.wav2spec(batch['recv_wav'])[..., 0].permute(0, 2, 1)
            tgt = self.wav2spec(batch['src_wav'])[..., 0].permute(0, 2, 1)

        img_feat = self.get_visual_feat(batch) if self.use_visual else None
        pred, _ = self.conformer(src, img_feat=img_feat)
        if self.args.decode_wav:
            pred = pred.squeeze(1)

        return {'pred': pred, 'tgt': tgt}


def dereverb_avspeech(batch, encode_wav, device, use_vida=False):
    with torch.no_grad():
        global DEREVERBERATOR
        if DEREVERBERATOR is None:
            DEREVERBERATOR = load_dereverberator(device, use_vida)
        pred_clean = DEREVERBERATOR.dereverb(batch, 0)['pred']
        if encode_wav:
            batch['src_wav'] = pred_clean[:, :batch['recv_wav'].shape[1]]
        else:
            batch['src_wav'] = pred_clean


def convolve_random_rir(batch):
    rirs = torch.flip(batch['rir'], (1,)).unsqueeze(1)
    outputs = []
    for i in range(batch['src_wav'].shape[0]):
        rir = rirs[i: i + 1]
        audio = batch['src_wav'].unsqueeze(1)[i:i + 1]
        outputs.append(torch.nn.functional.conv1d(audio, rir, padding=rir.shape[-1] - 1))
    outputs = torch.cat(outputs, dim=0).squeeze(1)
    batch['src_wav'] = outputs[:, :batch['recv_wav'].shape[1]].detach()
    batch['src_wav'] /= batch['src_wav'].abs().max(dim=1, keepdim=True)[0]
    batch['src_wav'] = torch.nan_to_num(batch['src_wav'])
    return batch


def apply_audio_data_augmentation(args, batch, phase):
    from torch_audiomentations import Compose, PolarityInversion, AddColoredNoise

    transforms = []
    transforms.append(AddColoredNoise(p=1, min_snr_in_db=args.min_snr, max_snr_in_db=args.max_snr))
    transforms.append(PolarityInversion(p=0.5))
    
    # Initialize augmentation callable
    if len(transforms) != 0:
        apply_augmentation = Compose(transforms=transforms)
        prev_shape = batch['src_wav'].shape
        batch['src_wav'] = apply_augmentation(batch['src_wav'].unsqueeze(1), sample_rate=16000).squeeze(1)
        assert batch['src_wav'].shape == prev_shape, (batch['src_wav'].shape, prev_shape)