# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torch.nn.functional as F
import speechmetrics

from vam.models.base_av_model import BaseAVModel
from vam.models.vida import unet_conv, unet_upconv, VisualNet, create_conv
from vam.models.avitar import dereverb_avspeech, convolve_random_rir, apply_audio_data_augmentation

DEREVERBERATOR = None


class UNET(BaseAVModel):
    def __init__(self, args):
        super(UNET, self).__init__(args)
        self.args = args
        input_nc = 2 if args.predict_mask else 1
        ngf = 64
        output_nc = 2 if args.predict_mask else 1
        self.use_visual = args.use_depth + args.use_rgb
        if self.use_visual:
            self.visual_encoder = VisualNet(torchvision.models.resnet18(pretrained=True),
                                            args.use_depth + args.use_rgb * 3)
            self.conv1x1 = create_conv(512, 8, 1, 0)

        if args.use_avspeech:
            if args.use_da:
                num_intermediate_feature = 832
            else:
                num_intermediate_feature = 992
        else:
            num_intermediate_feature = 512 + 864 if self.use_visual else 512

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(num_intermediate_feature,
                                                 ngf * 8)  # 1296  = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True)

        self.first_val = True
        self.best_val_loss = float('inf')
        self.configure_loss()
        self.metrics = ['speech_rt60', 'mag_distance', 'log_mag_distance', 'mosnet']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def forward(self, audio, visual):
        if visual is not None:
            visual_feat = self.conv1x1(self.visual_encoder(visual))

        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        upconv_feature_input = audio_conv5feature
        if visual is not None:
            visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
            visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2],
                                             audio_conv5feature.shape[-1])  # tile visual feature
            upconv_feature_input = torch.cat((visual_feat, upconv_feature_input), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(upconv_feature_input)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return prediction

    def acoustic_match(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        if self.args.dereverb_avspeech:
            dereverb_avspeech(batch, self.args.encode_wav, self.device)
        batch['original_src_wav'] = batch['src_wav']

        if self.args.convolve_random_rir:
            convolve_random_rir(batch)
            
        if self.args.use_audio_da:
            apply_audio_data_augmentation(self.args, batch, phase)

        src_spec = self.wav2spec(batch['src_wav']).permute(0, 3, 2, 1)
        tgt_spec = self.wav2spec(batch['recv_wav']).permute(0, 3, 2, 1)
        if not self.args.predict_mask:
            src_spec = src_spec[:, :1, ...]
            tgt_spec = tgt_spec[:, :1, ...]

        visual_inputs = []
        if self.args.use_depth:
            visual_inputs.append(batch['depth'])
        if self.args.use_rgb:
            visual_inputs.append(batch['rgb'])
        visual_inputs = torch.cat(visual_inputs, dim=1)
        pred = self.forward(src_spec, visual_inputs)
        src_spec_copy = src_spec.clone().detach()
        if self.args.predict_mask:
            src_spec_copy[:, :, 1:, :-1] *= pred
        else:
            src_spec_copy[:, :, 1:, :-1] = pred
        pred_spec = src_spec_copy
        assert pred_spec.shape == tgt_spec.shape, (pred_spec.shape, tgt_spec.shape)

        return {'pred': pred_spec, 'tgt': tgt_spec}

    def dereverb(self, batch, batch_idx):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        # inputs = batch['recv_spec'].permute(0, 3, 1, 2) # use two channel inputs
        recv_spec = self.wav2spec(batch['recv_wav']).permute(0, 3, 1, 2)
        tgt = self.wav2spec(batch['src_wav']).permute(0, 3, 1, 2)
        pred = self.forward(recv_spec[:, :1, :, :], visual=None)
        tgt = tgt[:, :1, :, :]
        full_mag_spec = torch.zeros_like(recv_spec[:, :1, :, :], device=self.device)
        full_mag_spec[:, :, :-1, :-1] = pred
        pred = full_mag_spec
        assert pred.shape == tgt.shape, (pred.shape, tgt.shape)

        pred = pred.permute(0, 1, 3, 2)
        tgt = tgt.permute(0, 1, 3, 2)

        return {'pred': pred, 'tgt': tgt}