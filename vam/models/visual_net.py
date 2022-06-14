# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torchvision


class VisualNet(nn.Module):
    def __init__(self, use_rgb, use_depth, use_audio):
        super(VisualNet, self).__init__()
        assert use_rgb or use_depth or use_audio
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_audio = use_audio

        in_channel = use_rgb * 3 + use_depth + use_audio
        conv1 = nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers = list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
        self.feature_extraction = nn.Sequential(conv1, *layers)  # features before conv1x1
        self.predictor = nn.Sequential(nn.Linear(512, 1))

    def forward(self, inputs):
        audio_feature = self.feature_extraction(inputs).squeeze(-1).squeeze(-1)
        pred = self.predictor(audio_feature)

        return pred
