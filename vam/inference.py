# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import glob
import os
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
from pytorch_lightning import seed_everything
seed_everything(1)
# torch.use_deterministic_algorithms(d=True)

from vam.trainer import parser
parser.add_argument('--input-image', type=str)
parser.add_argument('--input-audio', type=str)
parser.add_argument('--output-audio', type=str)


def main():
    args = parser.parse_args()
    assert args.acoustic_matching or args.dereverb

    args.slurm = False
    args.n_gpus = 1
    args.num_node = 1
    args.progress_bar = True
    args.batch_size = 16

    folder = args.model_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    from vam.models.generative_avitar import GenerativeAViTAr
    model = GenerativeAViTAr(args)

    if args.eval_last or (args.auto_resume and not args.test):
        existing_checkpoints = sorted(glob.glob(os.path.join(args.model_dir, args.version, f'avt_epoch=*.ckpt')))
        if len(existing_checkpoints) != 0:
            args.from_pretrained = existing_checkpoints[-1]
            print(args.from_pretrained)
        else:
            print('There is no existing checkpoints!')

    if args.eval_ckpt != -1:
        args.from_pretrained = os.path.join(args.model_dir, args.version, f'avt_epoch={args.eval_ckpt:04}.ckpt')
        print(args.from_pretrained)

    if args.eval_best:
        args.from_pretrained = os.path.join(args.model_dir, args.version, f'best_val.ckpt')
        print(args.from_pretrained)

    if os.path.exists(args.from_pretrained):
        model.load_weights(torch.load(args.from_pretrained, map_location='cpu'))
    else:
        print('Warning: no pretrained model weights are found!')
    model.to(device=torch.device('cuda'))
    model.eval()
    with torch.no_grad():
        rgb = np.array(Image.open(args.input_image))
        print(rgb.shape)
        audio, _ = librosa.load(args.input_audio, sr=16000)
        audio_length = 40960
        audio = audio[:audio_length]  # currently the model only takes audio input of length 2.56s
        transforms = [A.Resize(height=270, width=480),
                      A.CenterCrop(height=180, width=320),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                      ToTensorV2(),
        ]
        transform = A.Compose(transforms)
        batch = {'rgb': transform(image=rgb)['image'].unsqueeze(0),
                 'src_wav': torch.zeros(audio_length).unsqueeze(0),
                 'recv_wav': torch.from_numpy(audio).unsqueeze(0)
                 }
        print(batch['rgb'].shape, batch['recv_wav'].shape)
        output = model.acoustic_match(batch, 0, phase='test')
        pred, tgt = output['pred'], output['tgt']
        print(pred.shape, tgt.shape)
        sf.write(args.output_audio, pred.cpu().numpy()[0], 16000)


if __name__ == "__main__":
    main()
