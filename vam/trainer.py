# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shutil
import logging
import glob
import os
import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import torch
from tqdm import tqdm
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from vam.models.avitar import AViTAr
from vam.datasets.ss_speech_dataset import SoundSpacesSpeechDataset, to_tensor
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything
seed_everything(1)


parser = argparse.ArgumentParser()
parser.add_argument("--run-type", choices=["train", "eval"], default='train')
parser.add_argument("--model-dir", default='')
parser.add_argument("--version", default='v1')
parser.add_argument("--eval-best", default=False, action='store_true')
parser.add_argument("--eval-last", default=False, action='store_true')
parser.add_argument("--auto-resume", default=False, action='store_true')
parser.add_argument("--overwrite", default=False, action='store_true')
parser.add_argument("--use-rgb", default=False, action='store_true')
parser.add_argument("--use-depth", default=False, action='store_true')
parser.add_argument("--num-channel", default=1, type=int)
parser.add_argument("--n-gpus", default=1, type=int)
parser.add_argument("--num-node", default=1, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--max-epochs", default=600, type=int)
parser.add_argument("--ckpt-interval", default=10, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--generator-lr", default=0.0002, type=float)
parser.add_argument("--discriminator-lr", default=0.0002, type=float)
parser.add_argument("--wd", default=0, type=float)
parser.add_argument("--log-mag", default=False, action='store_true')
parser.add_argument("--no-mask", default=False, action='store_true')
parser.add_argument("--log1p", default=False, action='store_true')
parser.add_argument("--log10", default=False, action='store_true')
parser.add_argument("--save-ckpt-interval", default=1, type=int)
parser.add_argument("--limited-fov", default=False, action='store_true')
parser.add_argument("--from-pretrained", default='', type=str)
parser.add_argument("--slurm", default=False, action='store_true')
parser.add_argument("--gpu-mem32", default=False, action='store_true')
parser.add_argument("--part", default='learnlab,learnfair', type=str)
parser.add_argument("--num-encoder-layers", default=6, type=int)
parser.add_argument("--input-dropout-p", default=0.1, type=float)
parser.add_argument("--feed-forward-dropout-p", default=0.1, type=float)
parser.add_argument("--attention-dropout-p", default=0.1, type=float)
parser.add_argument("--conv-dropout-p", default=0.1, type=float)
parser.add_argument("--no-dropout", default=False, action='store_true')
parser.add_argument("--use-cnn", default=False, action='store_true')
parser.add_argument("--generate-reverb", default=False, action='store_true')
parser.add_argument("--acoustic-matching", default=False, action='store_true')
parser.add_argument("--dereverb", default=False, action='store_true')
parser.add_argument("--test", default=False, action='store_true')
parser.add_argument("--test-split", default='test-unseen', type=str)
parser.add_argument("--visualize", default=False, action='store_true')
parser.add_argument("--use-pretrained-rt60-predictor", default=False, action='store_true')
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--loss", default='mse', type=str)
parser.add_argument("--fast-dev-run", default=False, action='store_true')
parser.add_argument("--progress-bar", default=False, action='store_true')
parser.add_argument("--remove-oov", default=False, action='store_true')
parser.add_argument("--eval-ckpt", default=-1, type=int)
parser.add_argument("--use-avspeech", default=False, action='store_true')
parser.add_argument("--dereverb-avspeech", default=False, action='store_true')
parser.add_argument("--conv-kernel-size", default=31, type=int)
parser.add_argument("--pretrained-cnn", default=False, action='store_true')
parser.add_argument("--model", default='avitar', type=str)
parser.add_argument("--decode-wav", default=False, action='store_true')
parser.add_argument("--encode-wav", default=False, action='store_true')
parser.add_argument("--pretrained-transformer", default='', type=str)
parser.add_argument("--freeze-transformer", default=False, action='store_true')
parser.add_argument("--hop-length", default=160, type=int)
parser.add_argument("--norm-rir-len", default=False, action='store_true')
parser.add_argument("--multires-stft", default=False, action='store_true')
parser.add_argument("--remove-mel-loss", default=False, action='store_true')
parser.add_argument("--fft-sizes", default='256,512,1024', type=str)
parser.add_argument("--hop-sizes", default='64,128,256', type=str)
parser.add_argument("--win-lengths", default='128,400,600', type=str)
parser.add_argument("--stft-loss-weight", default=45, type=int)
parser.add_argument("--encoder-ratios", default='8,4,2,2', type=str)
parser.add_argument("--decoder-ratios", default='8,4,2,2', type=str)
parser.add_argument("--encoder-residual-layers", default=3, type=int)
parser.add_argument("--decoder-residual-layers", default=3, type=int)
parser.add_argument("--eval-speech-rt60", default=True, action='store_true')
parser.add_argument("--use-librispeech", default=False, action='store_true')
parser.add_argument("--remove-fm-loss", default=False, action='store_true')
parser.add_argument("--val-freq", default=10, type=int)
parser.add_argument("--ngf", default=32, type=int)
parser.add_argument("--use-visual-pe", default=False, action='store_true')
parser.add_argument("--predict-mask", default=False, action='store_true')
parser.add_argument("--use-real-imag", default=False, action='store_true')
parser.add_argument("--num-worker", default=2, type=int)
parser.add_argument("--convolve-random-rir", default=False, action='store_true')
parser.add_argument("--use-da", default=False, action='store_true')
parser.add_argument("--eval-input", default=False, action='store_true')
parser.add_argument("--use-vida", default=False, action='store_true')
parser.add_argument("--use-audio-da", default=False, action='store_true')
parser.add_argument("--min-snr", default=2, type=float)
parser.add_argument("--max-snr", default=10, type=float)
parser.add_argument("--noise-first", default=False, action='store_true')
parser.add_argument("--read-mp4", default=False, action='store_true')
parser.add_argument("--eval-clean", default=False, action='store_true')
parser.add_argument("--adaptive-pool", default=False, action='store_true')
parser.add_argument("--mean-pool", default=False, action='store_true')
parser.add_argument("--apply-high-pass", default=False, action='store_true')
parser.add_argument("--crossmodal-dropout-p", default=-1, type=float)
parser.add_argument("--comment", default="", type=str)
parser.add_argument("--prev-ckpt", default=-1, type=int)
parser.add_argument("--generate-plot", default=False, action='store_true')
parser.add_argument("--save-features", default=False, action='store_true')


def main():
    args = parser.parse_args()

    assert args.acoustic_matching or args.dereverb

    if args.model_dir == '':
        if args.acoustic_matching:
            if args.use_avspeech:
                args.model_dir = 'data/models/vam/avspeech'
            else:
                args.model_dir = 'data/models/vam/am'
        else:
            args.model_dir = 'data/models/vam/dereverb'
        print(f'Model dir: {args.model_dir}')

    if args.test or args.visualize or args.fast_dev_run:
        args.slurm = False
        args.n_gpus = 1
        args.num_node = 1
        args.progress_bar = True
        args.batch_size = 16

    if args.test:
        args.batch_size = 32
        args.num_worker = 10

    if args.fast_dev_run:
        if os.path.exists('data/models/debug'):
            shutil.rmtree('data/models/debug')
        args.model_dir = 'data/models/debug'
        args.version = 'v1'
        args.batch_size = 2

    if args.eval_clean:
        assert args.use_avspeech
        args.dereverb_avspeech = False
        args.use_librispeech = True

    folder = args.model_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    if args.slurm:
        import submitit
        executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
        executor.update_parameters(slurm_job_name=args.version, timeout_min=60*72,
                                   slurm_partition=args.part, nodes=args.num_node, gpus_per_node=args.n_gpus, cpus_per_task=10,
                                   slurm_constraint='volta32gb' if args.gpu_mem32 else None, slurm_mem=70 * 1024,
                                   tasks_per_node=args.n_gpus, comment=args.comment
                                   )
        job = executor.submit(run, args)
        print(job.job_id)
    else:
        run(args)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.use_avspeech:
        from vam.datasets.avspeech_dataset import AVSpeechDataset
        dataset = AVSpeechDataset
    else:
        dataset = SoundSpacesSpeechDataset
    if args.test or args.visualize:
        test_set = dataset(split=args.test_split, use_rgb=True, use_depth=True, hop_length=args.hop_length,
                           remove_oov=args.remove_oov, use_librispeech=args.use_librispeech,
                           convolve_random_rir=args.convolve_random_rir, use_da=args.use_da,
                           read_mp4=args.read_mp4)
        test_dataset = torch.utils.data.DataLoader(test_set, num_workers=10, batch_size=args.batch_size, pin_memory=True)
    else:
        train_set = dataset(split='train', use_rgb=True, use_depth=True, hop_length=args.hop_length,
                            remove_oov=args.remove_oov, use_librispeech=args.use_librispeech,
                            convolve_random_rir=args.convolve_random_rir, use_da=args.use_da,
                            read_mp4=args.read_mp4)
        train_dataset = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=args.num_worker, pin_memory=True,
                                                    batch_size=args.batch_size)
        val_set = dataset(split='val', use_rgb=True, use_depth=True, hop_length=args.hop_length,
                          remove_oov=args.remove_oov, convolve_random_rir=args.convolve_random_rir, use_da=args.use_da,
                          read_mp4=args.read_mp4)
        val_dataset = torch.utils.data.DataLoader(val_set, num_workers=args.num_worker, batch_size=args.batch_size, pin_memory=True)

    if args.model == 'avitar':
        model = AViTAr(args)
    elif args.model == 'generative_avitar':
        from vam.models.generative_avitar import GenerativeAViTAr
        model = GenerativeAViTAr(args)
    elif args.model == 'unet':
        from vam.models.unet import UNET
        model = UNET(args)
    elif args.model == 'blind_reverberator':
        from vam.models.blind_reverberator import BlindReverberator
        model = BlindReverberator(args)
    else:
        raise ValueError

    # Model training
    logger = loggers.TensorBoardLogger(
        args.model_dir,
        version=args.version
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.model_dir, args.version),
        filename="vam_{epoch:04d}",
        every_n_val_epochs=args.ckpt_interval,
        save_top_k=-1,
        verbose=True,
    )

    if args.eval_last or (args.auto_resume and not args.test):
        existing_checkpoints = sorted(glob.glob(os.path.join(args.model_dir, args.version, f'vam_epoch=*.ckpt')))
        if len(existing_checkpoints) != 0:
            args.from_pretrained = existing_checkpoints[-1]
            print(args.from_pretrained)
        else:
            print('There is no existing checkpoints!')

    if args.eval_ckpt != -1:
        args.from_pretrained = os.path.join(args.model_dir, args.version, f'vam_epoch={args.eval_ckpt:04}.ckpt')
        print(args.from_pretrained)

    if args.eval_best:
        args.from_pretrained = os.path.join(args.model_dir, args.version, f'best_val.ckpt')
        print(args.from_pretrained)

    trainer = Trainer(
        gpus=args.n_gpus,
        num_nodes=args.num_node,
        accelerator="ddp",
        benchmark=True,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=args.from_pretrained,
        default_root_dir=args.model_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
        progress_bar_refresh_rate=args.fast_dev_run or args.progress_bar,
        fast_dev_run=args.fast_dev_run,
        plugins=DDPPlugin(find_unused_parameters=False)
    )

    if not args.test and not args.visualize:
        trainer.fit(model, train_dataset, val_dataset)
    elif args.test:
        assert True
        model.load_weights(torch.load(args.from_pretrained, map_location='cpu'))
        trainer.test(model, test_dataloaders=test_dataset)
        model.save_test_stats()
    elif args.visualize:
        assert True
        model.load_weights(torch.load(args.from_pretrained, map_location='cpu'))
        model.to(device=torch.device('cuda'))
        model.eval()
        with torch.no_grad():
            generate_qual(args, model, test_dataset)
    else:
        raise ValueError


def generate_qual(args, model, dataset):
    import matplotlib.pyplot as plt
    import soundfile as sf
    from librosa.display import waveplot

    count = 0
    for i, batch in enumerate(dataset):
        if args.acoustic_matching:
            output = model.acoustic_match(batch, i, phase='test')
        else:
            output = model.dereverb(batch, i)
        pred, tgt = output['pred'], output['tgt']

        pred_wav = pred if args.decode_wav else model.spec2wav(pred)
        input_wav = batch['original_src_wav'] if args.acoustic_matching else batch['recv_wav']
        tgt_wav = batch['recv_wav'] if args.acoustic_matching else batch['src_wav']
        input_spec = torch.stft(input_wav.squeeze(), n_fft=512, hop_length=160, win_length=400,
                               window=torch.hamming_window(400, device=pred_wav.device), pad_mode='constant',
                               center=True, return_complex=True).abs()
        pred_spec = torch.stft(pred_wav.squeeze(), n_fft=512, hop_length=160, win_length=400,
                               window=torch.hamming_window(400, device=pred_wav.device), pad_mode='constant',
                               center=True, return_complex=True).abs()
        tgt_spec = torch.stft(tgt_wav.squeeze(), n_fft=512, hop_length=160, win_length=400,
                              window=torch.hamming_window(400, device=pred_wav.device), pad_mode='constant',
                              center=True, return_complex=True).abs()
        input_wav, pred_wav, tgt_wav, pred_spec, tgt_spec, input_spec = \
            input_wav.cpu(), pred_wav.cpu(), tgt_wav.cpu(), pred_spec.cpu(), tgt_spec.cpu(), input_spec.cpu()

        # plot log spec
        input_spec = torch.log1p(input_spec)
        tgt_spec = torch.log1p(tgt_spec)
        pred_spec = torch.log1p(pred_spec)

        plots_dir = os.path.join(args.model_dir, args.version, f"plots_{args.test_split}{'_clean' if args.eval_clean else ''}")
        audio_dir = os.path.join(args.model_dir, args.version, f"audio_{args.test_split}{'_clean' if args.eval_clean else ''}")
        image_dir = os.path.join(args.model_dir, args.version, f"images_{args.test_split}")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        input_spec, input_wav, tgt_spec, pred_spec, tgt_wav, pred_wav = \
            input_spec.numpy(), input_wav.numpy(), tgt_spec.numpy(), pred_spec.numpy(), tgt_wav.numpy(), pred_wav.numpy()
        print(tgt_spec.shape, pred_spec.shape, tgt_wav.shape, pred_wav.shape)

        # for better visual effect
        input_spec = input_spec[:, :input_spec.shape[1]//2, :]
        tgt_spec = tgt_spec[:, :tgt_spec.shape[1]//2, :]
        pred_spec = pred_spec[:, :pred_spec.shape[1]//2, :]
        for j in tqdm(range(pred_wav.shape[0])):
            if args.generate_plot:
                fig, axes = plt.subplots(2, 3)
                axes[0][0].imshow(input_spec[j])
                axes[0][0].set_aspect(1 / axes[0][0].get_data_ratio())
                axes[0][0].invert_yaxis()
                axes[0][0].set_xticks([])
                axes[0][0].set_yticks([])

                plt.sca(axes[1][0])
                waveplot(input_wav[j], 16000)
                axes[1][0].set_xticks([])
                axes[1][0].set_yticks([])

                axes[0][1].imshow(pred_spec[j])
                axes[0][1].set_aspect(1 / axes[0][1].get_data_ratio())
                axes[0][1].invert_yaxis()
                axes[0][1].set_xticks([])
                axes[0][1].set_yticks([])

                plt.sca(axes[1][1])
                waveplot(pred_wav[j], 16000)
                axes[1][1].set_xticks([])
                axes[1][1].set_yticks([])

                axes[0][2].imshow(tgt_spec[j])
                axes[0][2].set_aspect(1 / axes[0][2].get_data_ratio())
                axes[0][2].invert_yaxis()
                axes[0][2].set_xticks([])
                axes[0][2].set_yticks([])

                plt.sca(axes[1][2])
                waveplot(tgt_wav[j], 16000)
                axes[1][2].set_xticks([])
                axes[1][2].set_yticks([])

                fig.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{count}.png'))
                plt.close(fig)

                if 'rgb' in batch:
                    rgb = batch['rgb'] if not args.use_da else batch['original_rgb']
                    rgb = rgb.cpu()[j].permute(1, 2, 0).numpy()
                    plt.imsave(os.path.join(image_dir, f'{count}-rgb.png'), rgb)
                if 'depth' in batch:
                    depth = batch['depth'].cpu()[j][0].numpy()
                    plt.imsave(os.path.join(image_dir, f'{count}-depth.png'), depth)

            sf.write(os.path.join(audio_dir, f'{count}-tgt.wav'), tgt_wav[j], samplerate=16000)
            sf.write(os.path.join(audio_dir, f'{count}-pred.wav'), pred_wav[j], samplerate=16000)
            sf.write(os.path.join(audio_dir, f'{count}-input.wav'), input_wav[j], samplerate=16000)
            count += 1


if __name__ == "__main__":
    main()
