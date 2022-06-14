# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import glob
from tqdm import tqdm

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class SoundSpacesSpeechDataset(Dataset):
    def __init__(self, split, normalize_whole=True, normalize_segment=False,
                 use_real_imag=False, use_rgb=False, use_depth=False, use_seg=False, limited_fov=False, crop=False,
                 flip=False, use_rgbd=False, remove_oov=False, hop_length=160,
                 deterministic_eval=False, use_librispeech=False, convolve_random_rir=False, use_da=False,
                 read_mp4=False, use_recv_audio=False):
        self.split = split
        self.data_dir = os.path.join('data/soundspaces_speech', split)
        self.files = glob.glob(self.data_dir + '/**/*.pkl', recursive=True)
        np.random.shuffle(self.files)
        self.normalize_whole = normalize_whole
        self.normalize_segment = normalize_segment
        self.use_real_imag = use_real_imag
        self.use_rgb = use_rgb or use_rgbd
        self.use_depth = use_depth or use_rgbd
        self.use_seg = use_seg
        self.limited_fov = limited_fov
        self.crop = crop
        self.flip = flip
        self.hop_length = hop_length
        self.deterministic_eval = deterministic_eval

        if remove_oov:
            tgt_file = os.path.join('data/soundspaces_speech/remove_oov', split + '.pkl')
            if os.path.exists(tgt_file):
                with open(tgt_file, 'rb') as fo:
                    self.files = pickle.load(fo)
                print(f'Read from file {tgt_file}')
            else:
                new_files = []
                for file in tqdm(self.files):
                    with open(file, 'rb') as fo:
                        file_data = pickle.load(fo)
                    speaker_in_view = np.sum(
                        np.concatenate([x == 41 for x in file_data['semantic']], axis=1)) >= 50
                    if speaker_in_view and np.argmax(file_data['rir']) == 0:
                        new_files.append(file)
                self.files = new_files
                with open(tgt_file, 'wb') as fo:
                    pickle.dump(new_files, fo)
                    print(f'Write to file {tgt_file}')
            print(f"Number of files after removing out-of-view cases: {len(self.files)}")

    def __len__(self):
        # return len(self.files)
        # TODO: need to fix the length
        return 100

    def __getitem__(self, item):
        file = self.files[item]
        with open(file, 'rb') as fo:
            data = pickle.load(fo)

        receiver_audio = data['receiver_audio']
        source_audio = data['source_audio']
        location = data['location']
        rir = data['rir']
        distance = data['geodesic_distance']

        src_wav, recv_wav = self.process_audio(source_audio, receiver_audio)

        sample = dict()
        sample['src_wav'] = src_wav
        sample['recv_wav'] = recv_wav
        rir_wav = process_rir(rir)
        sample['rir_wav'] = rir_wav

        # stitch images into panorama
        visual_sensors = []
        if self.use_rgb:
            sample['rgb'] = to_tensor(np.concatenate([x / 255.0 for x in data['rgb']], axis=1)).permute(2, 0, 1)
            visual_sensors.append('rgb')
        if self.use_depth:
            sample['depth'] = to_tensor(np.concatenate(data['depth'], axis=1)).unsqueeze(0)
            visual_sensors.append('depth')
        if self.use_seg:
            sample['seg'] = to_tensor(np.concatenate([x == 41 for x in data['semantic']], axis=1)).unsqueeze(0)
            visual_sensors.append('seg')

        if len(visual_sensors) > 0:
            if self.split == 'train':
                # data augmentation
                width_shift = None
                if self.flip:
                    is_flipped = np.random.uniform()
                for visual_sensor in visual_sensors:
                    if width_shift is None:
                        width_shift = np.random.randint(0, sample[visual_sensor].shape[-1])
                    sample[visual_sensor] = torch.roll(sample[visual_sensor], width_shift, dims=-1)
                    if self.flip and is_flipped:
                        sample[visual_sensor] = torch.flip(sample[visual_sensor], dims=(2,))

            if self.limited_fov:
                # crop image to size 384 * 256
                if self.split == 'train':
                    offset = None
                    for visual_sensor in visual_sensors:
                        if offset is None:
                            offset = np.random.randint(0, sample[visual_sensor].shape[-1] - 256)
                        sample[visual_sensor] = sample[visual_sensor][:, :, offset: offset + 256]
                else:
                    for visual_sensor in visual_sensors:
                        sample[visual_sensor] = sample[visual_sensor][:, :, :256]

                # crop images
                if self.crop:
                    for visual_sensor in visual_sensors:
                        sample[visual_sensor] = torchvision.transforms.RandomCrop((336, 224))(sample[visual_sensor])
            else:
                for visual_sensor in visual_sensors:
                    if self.crop:
                        sample[visual_sensor] = torchvision.transforms.RandomCrop((336, 1008))(sample[visual_sensor])
                        sample[visual_sensor] = torchvision.transforms.Resize((168, 504))(sample[visual_sensor])
                    else:
                        sample[visual_sensor] = torchvision.transforms.Resize((192, 576))(sample[visual_sensor])

        return sample
    
    def process_audio(self, source_audio, receiver_audio, use_start=False):
        # normalize the intensity before padding
        waveform_length = 40960

        if self.normalize_whole:
            receiver_audio = normalize(receiver_audio, norm='peak')
            source_audio = normalize(source_audio, norm='peak')

        # pad audio
        if waveform_length > source_audio.shape[0]:
            receiver_audio = np.pad(receiver_audio, (0, max(0, waveform_length - receiver_audio.shape[0])))
            source_audio = np.pad(source_audio, (0, waveform_length - source_audio.shape[0]))

        if (self.deterministic_eval and self.split != 'train') or use_start:
            start_index = 0
        else:
            start_index = np.random.randint(0, source_audio.shape[0] - waveform_length) \
                            if source_audio.shape[0] != waveform_length else 0
        source_clip = source_audio[start_index: start_index + waveform_length]
        receiver_clip = receiver_audio[start_index: start_index + waveform_length]

        # normalize the short clip
        if self.normalize_segment:
            source_clip = normalize(source_clip, norm='peak')
            receiver_clip = normalize(receiver_clip, norm='peak')

        return source_clip, receiver_clip


def compute_material(seg):
    seg = to_tensor(np.concatenate(seg, axis=1)).unsqueeze(0)
    material_dist = np.zeros(42)
    for i in range(42):
        material_dist[i] = (seg == i).sum()
    material_dist /= seg.shape[1] * seg.shape[2]
    return material_dist


def normalize(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError


def process_rir(rir):
    # normalize the intensity before padding
    rir = normalize(np.trim_zeros(rir, 'fb'), norm='peak')
    max_length = 17536

    # pad audio
    if max_length > rir.shape[0]:
        rir = np.pad(rir, (0, max(0, max_length - rir.shape[0])))

    rir = to_tensor(rir)
    return rir


def calculate_drr_energy_ratio(y, cutoff_time, sr=16000):
    direct_sound_idx = int(cutoff_time * sr)

    # removing leading silence
    y = normalize(y)
    y = np.trim_zeros(y, trim='fb')

    # everything up to the given idx is summed up and treated as direct sound energy
    y = np.power(y, 2)
    direct = sum(y[:direct_sound_idx + 1])
    reverberant = sum(y[direct_sound_idx + 1:])
    if direct == 0 or reverberant == 0:
        drr = 1
        print('Direct or reverberant is 0')
    else:
        drr = 10 * np.log10(direct / reverberant)

    return drr


def measure_decay_time(h, fs=1, decay_db=60):
    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    return decay_time