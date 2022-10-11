# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import pickle

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from PIL import Image
from scipy.io import wavfile

from vam.datasets.ss_speech_dataset import to_tensor
from vam.datasets.ss_speech_dataset import SoundSpacesSpeechDataset


class AVSpeechDataset(SoundSpacesSpeechDataset):
    def __init__(self, split, normalize_whole=True, normalize_segment=False, use_real_imag=False,
                 use_rgb=False, use_depth=False, limited_fov=False,
                 remove_oov=False, hop_length=160, use_librispeech=False, convolve_random_rir=False, use_da=False,
                 read_mp4=False):
        super().__init__(split)
        self.split = split
        self.normalize_whole = normalize_whole
        self.normalize_segment = normalize_segment
        self.use_real_imag = use_real_imag
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.limited_fov = limited_fov
        self.hop_length = hop_length
        self.rgb_res = (180, 320)
        self.use_librispeech = use_librispeech
        self.convolve_random_rir = convolve_random_rir
        self.use_da = use_da
        self.read_mp4 = read_mp4

        self.data_dir = 'data/acoustic_avspeech'
        files = sorted(os.listdir(os.path.join(self.data_dir, 'img')))

        if split == 'train':
            self.files = files[: int(len(files) * 0.95)]
        elif split == 'test-seen':
            self.files = files[: int(len(files) * 0.025)]
        elif split == 'val':
            self.files = files[int(len(files) * 0.95) + 1: int(len(files) * 0.975)]
        else:
            self.files = files[int(len(files) * 0.975) + 1:]

        if use_librispeech:
            speech_split = split if split != 'test-seen' else 'train'
            self.speech_files = sorted(glob.glob(f'data/soundspaces_speech/{speech_split}/**/*.pkl', recursive=True))

        if self.convolve_random_rir:
            rir_dict_file = 'data/acoustic_avspeech/random_rir.pkl'
            assert os.path.exists(rir_dict_file)
            with open(rir_dict_file, 'rb') as fo:
                rir_split = split if split != 'test-seen' else 'train'
                self.rir_list = pickle.load(fo)[rir_split]
                print(f'Number of rirs: {len(self.rir_list)}')

        if use_da:
            # 180, 320, 144, 256
            transforms = [A.Resize(height=270, width=480)] if self.read_mp4 else []
            if split == 'train':
                transforms += [
                        A.RandomCrop(height=180, width=320),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),
                    ]
            else:
                transforms += [
                        A.CenterCrop(height=180, width=320),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),
                    ]
            self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_file = os.path.join(self.data_dir, 'img', self.files[item])
        audio_file = img_file.replace('img/', 'audio/').replace('.png', '.wav')
        rgb = np.array(Image.open(img_file))
        sr, recv_audio = wavfile.read(audio_file)
        src_audio = np.zeros_like(recv_audio)

        if self.use_librispeech:
            speech_file = self.speech_files[item % len(self.speech_files)]
            with open(speech_file, 'rb') as fo:
                speech_data = pickle.load(fo)
            src_audio = speech_data['source_audio']

        if rgb.shape[:2] != self.rgb_res:
            rgb = torchvision.transforms.Resize(self.rgb_res)(to_tensor(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        rgb = rgb.astype(np.float32)

        sample = dict()
        src_wav, recv_wav = self.process_audio(src_audio, recv_audio)
        sample['src_wav'] = src_wav
        sample['recv_wav'] = recv_wav
        if self.use_da:
            sample['original_rgb'] = to_tensor(rgb).permute(2, 0, 1) / 255.0
            sample['rgb'] = self.transform(image=rgb)['image']
        else:
            sample['rgb'] = to_tensor(rgb).permute(2, 0, 1) / 255.0

        if self.convolve_random_rir:
            max_rir_len = 24000
            # rir = np.random.choice(self.rir_list)
            rir = self.rir_list[item % len(self.rir_list)]
            padded_rir = np.pad(rir, (0, max(0, max_rir_len - rir.shape[0])))
            sample['rir'] = to_tensor(padded_rir)

        return sample